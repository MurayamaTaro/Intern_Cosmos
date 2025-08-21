"""
Cosmos-Predict1 (diffusion, text-to-video, 7B) 推論スクリプト（単一LoRA前提）。

- LoRA を使わないときは LoRA 関連の探索・準備を完全にスキップ。
- LoRA 変換テンポラリのメモリ解放を明示 (del + gc.collect)。
"""

import argparse
import os
import subprocess
import sys
import re
import shutil
import gc
from pathlib import Path
import torch

# ------------------------------
# チェックポイント探索 (単一実験)
# ------------------------------
def find_latest_lora_checkpoint_single_experiment(experiment_dir: Path) -> Path | None:
    """
    単一実験ディレクトリから最新のLoRA重みのパスを返す。
    優先: checkpoints/latest_checkpoint.txt -> <base>_model.pt
    代替: checkpoints/iter_*.pt の最大イテレーション -> <base>_model.pt
    """
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"Error: Checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        return None

    latest_txt = ckpt_dir / "latest_checkpoint.txt"
    if latest_txt.exists():
        try:
            base = latest_txt.read_text().strip().removesuffix(".pt")
            candidate = ckpt_dir / f"{base}_model.pt"
            if candidate.exists():
                print(f"Found LoRA checkpoint via latest_checkpoint.txt: {candidate}")
                return candidate
            else:
                print(f"Warning: {candidate} not found though latest_checkpoint.txt exists.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to read latest_checkpoint.txt: {e}", file=sys.stderr)

    # フォールバック: iter_*.pt を走査
    iter_files = [p for p in ckpt_dir.glob("iter_*.pt") if re.fullmatch(r"iter_\\d+\\.pt", p.name)]
    if not iter_files:
        print(f"Error: No iter_*.pt files found in {ckpt_dir}", file=sys.stderr)
        return None

    latest_manifest = max(iter_files, key=lambda p: int(re.search(r"iter_(\\d+)\\.pt", p.name).group(1)))
    lora_model = latest_manifest.with_name(f"{latest_manifest.stem}_model.pt")
    if not lora_model.exists():
        print(f"Error: Corresponding model weights not found: {lora_model}", file=sys.stderr)
        return None

    print(f"Found LoRA checkpoint via iter scan: {lora_model}")
    return lora_model


# LoRA 推論
def run_inference_for_lora_stage(
    lora_model_file: Path,
    prompt: str,
    output_base_dir: Path,
    num_videos: int,
    nproc_per_node: int,
    num_steps: int,
    fps: int,
    guidance: float,
):
    """
    LoRA 重みを使ってバッチ推論を実行。
    - Cosmos の text2world.py が期待するチェックポイント構成に一時変換
    - 大量コピーは避けて symlink を使用
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print("\\n" + "=" * 80)
    print(f"Running inference (LoRA): {output_base_dir.name}")
    print(f"Using LoRA file: {lora_model_file}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    tmp_checkpoint_dir = output_base_dir.parent / f"tmp_{output_base_dir.name}"
    compatible_lora_checkpoint_path = None
    try:
        if tmp_checkpoint_dir.exists():
            shutil.rmtree(tmp_checkpoint_dir)
        tmp_checkpoint_dir.mkdir(parents=True)

        # 1) 互換な LoRA チェックポイントを準備（GPU を使わないよう map_location='cpu'）
        print("Preparing compatible LoRA checkpoint...")
        original_checkpoint = torch.load(lora_model_file, map_location="cpu")
        lora_state_dict = original_checkpoint.get("model")
        if lora_state_dict is None:
            raise KeyError("Checkpoint does not contain a 'model' key.")

        compatible_checkpoint = {"model": lora_state_dict, "ema": {}}
        compatible_lora_checkpoint_path = tmp_checkpoint_dir / "compatible_model.pt"
        torch.save(compatible_checkpoint, compatible_lora_checkpoint_path)
        print(f"Compatible LoRA checkpoint saved to: {compatible_lora_checkpoint_path}")

        # 使い終わったテンソルは解放（念のため）
        del lora_state_dict
        del original_checkpoint
        gc.collect()

        # 2) 依存コンポーネントのリンク
        print("Preparing a temporary directory with symbolic links...")
        workspace_root = Path.cwd().resolve()
        original_checkpoints_root = workspace_root / "checkpoints"

        required_components = [
            "Cosmos-Tokenize1-CV8x8x8-720p",
            "google-t5",
        ]
        for name in required_components:
            src = original_checkpoints_root / name
            dst = tmp_checkpoint_dir / name
            if not src.exists():
                print(f"Warning: Required component '{src}' not found. Skipping.", file=sys.stderr)
                continue
            os.symlink(src, dst)
            print(f"Created symlink for '{name}'.")

        # 3) LoRA モデル用のディレクトリを作って model.pt へリンク
        model_type_subdir = "Cosmos-Predict1-7B-Text2World_post-trained-lora"
        nested_model_dir = tmp_checkpoint_dir / model_type_subdir
        nested_model_dir.mkdir(parents=True, exist_ok=True)
        symlink_model_path = nested_model_dir / "model.pt"
        os.symlink(compatible_lora_checkpoint_path, symlink_model_path)
        print(f"Created symlink for model: {symlink_model_path} -> {compatible_lora_checkpoint_path}")

        # 4) 生成ループ（シードを 0..num_videos-1）
        for i in range(num_videos):
            seed = i
            seed_output_dir = output_base_dir / f"seed_{seed}"
            seed_output_dir.mkdir(exist_ok=True)

            print(f"  Generating video for seed {seed}...")

            command = [
                "torchrun", f"--nproc_per_node={nproc_per_node}",
                "cosmos_predict1/diffusion/inference/text2world.py",
                "--num_gpus", str(nproc_per_node),
                "--checkpoint_dir", str(tmp_checkpoint_dir),
                "--diffusion_transformer_dir", model_type_subdir,
                "--prompt", prompt,
                "--num_steps", str(num_steps),
                "--video_save_folder", str(seed_output_dir),
                "--fps", str(fps),
                "--seed", str(seed),
                "--guidance", str(guidance),
                "--disable_guardrail",
                "--disable_prompt_upsampler",
            ]

            new_env = os.environ.copy()
            new_env["TRANSFORMERS_CACHE"] = str(original_checkpoints_root)

            process = subprocess.Popen(
                command,
                env=new_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            # 生成ファイル整理
            vids = list(seed_output_dir.glob("*.mp4"))
            if not vids:
                raise FileNotFoundError("Inference script did not produce an MP4 file.")
            src = vids[0]
            dst = seed_output_dir / "video.mp4"
            src.rename(dst)

            with open(seed_output_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            print(f"  Successfully generated: {dst}")

    except subprocess.CalledProcessError as e:
        i = locals().get("i", "N/A")
        print(f"\\n  Error: Inference process failed with return code {e.returncode} for seed {i}.", file=sys.stderr)
    except Exception as e:
        print(f"\\n  An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # テンポラリの掃除
        if "tmp_checkpoint_dir" in locals() and tmp_checkpoint_dir.exists():
            print(f"Cleaning up temporary directory: {tmp_checkpoint_dir}")
            shutil.rmtree(tmp_checkpoint_dir)
        gc.collect()


# ベースモデル推論
def run_inference_for_base_model(
    prompt: str,
    output_base_dir: Path,
    num_videos: int,
    nproc_per_node: int,
    num_steps: int,
    fps: int,
    guidance: float,
):
    """
    ベースモデルでのバッチ推論。
    text2world.py の CLI に合わせ、直接スクリプトパスで起動。
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print("\\n" + "=" * 80)
    print(f"Running inference (base model): {output_base_dir.name}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    workspace_root = Path.cwd().resolve()
    original_checkpoints_root = workspace_root / "checkpoints"

    for i in range(num_videos):
        seed = i
        seed_output_dir = output_base_dir / f"seed_{seed}"
        seed_output_dir.mkdir(exist_ok=True)

        print(f"  Generating video for seed {seed}...")

        command = [
            "torchrun", f"--nproc_per_node={nproc_per_node}",
            "cosmos_predict1/diffusion/inference/text2world.py",
            "--num_gpus", str(nproc_per_node),
            "--checkpoint_dir", str(original_checkpoints_root),
            "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Text2World",
            "--prompt", prompt,
            "--num_steps", str(num_steps),
            "--video_save_folder", str(seed_output_dir),
            "--fps", str(fps),
            "--seed", str(seed),
            "--guidance", str(guidance),
            "--disable_guardrail",
            "--disable_prompt_upsampler",
        ]

        new_env = os.environ.copy()
        new_env["TRANSFORMERS_CACHE"] = str(original_checkpoints_root)

        try:
            process = subprocess.Popen(
                command,
                env=new_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            vids = list(seed_output_dir.glob("*.mp4"))
            if not vids:
                raise FileNotFoundError("Inference script did not produce an MP4 file.")
            src = vids[0]
            dst = seed_output_dir / "video.mp4"
            src.rename(dst)

            with open(seed_output_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            print(f"  Successfully generated: {dst}")

        except subprocess.CalledProcessError as e:
            print(f"\\n  Error: Inference process failed with return code {e.returncode} for seed {seed}.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\\n  An unexpected error occurred for seed {seed}: {e}", file=sys.stderr)
            continue


# ------------------------------
# メイン
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run inference using the base model and/or a single LoRA for Cosmos-Predict1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference_name", type=str, required=True,
                        help="Name for this inference run (used under ./outputs).")
    parser.add_argument("--experiment_name", type=str,
                        help="Run name created by posttrain_single.py (required when running LoRA stage).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for video generation.")
    parser.add_argument("--num_videos", type=int, default=1,
                        help="Number of videos to generate (different seeds).")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="Number of GPUs for inference.")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second.")
    parser.add_argument("--guidance", type=float, default=8.0,
                        help="Guidance scale.")
    parser.add_argument("--stages", type=str, nargs="+",
                        default=["original", "lora"],
                        choices=["original", "lora", "both"],
                        help="Choose 'original', 'lora', or 'both'.")

    args = parser.parse_args()

    # ステージ正規化
    requested = set(args.stages)
    if "both" in requested:
        requested = {"original", "lora"}

    # 出力ルートは ./outputs
    workspace_root = Path.cwd().resolve()
    outputs_root = workspace_root / "outputs" / args.inference_name

    # --- Base Model Inference ---
    if "original" in requested:
        base_output_dir = outputs_root / "original"
        if base_output_dir.exists():
            print(f"Base model output directory '{base_output_dir}' already exists. Skipping base model inference.")
        else:
            run_inference_for_base_model(
                prompt=args.prompt,
                output_base_dir=base_output_dir,
                num_videos=args.num_videos,
                nproc_per_node=args.nproc_per_node,
                num_steps=args.num_steps,
                fps=args.fps,
                guidance=args.guidance,
            )
    else:
        print("Skipping base model inference.")

    # --- LoRA Model Inference ---
    if "lora" in requested:
        if not args.experiment_name:
            print("Error: --experiment_name is required when running the 'lora' stage.", file=sys.stderr)
            sys.exit(1)

        # posttrain_single.py の配置規約:
        # checkpoints/posttraining/diffusion_text2world/text2world_7b_lora_my/{experiment_name}
        experiment_dir = (
            workspace_root / "checkpoints" / "posttraining" / "diffusion_text2world"
            / "text2world_7b_lora_my" / args.experiment_name
        )
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found at {experiment_dir}", file=sys.stderr)
            sys.exit(1)

        lora_output_root = outputs_root / args.experiment_name
        vehicle_lora_file = find_latest_lora_checkpoint_single_experiment(experiment_dir)
        if vehicle_lora_file:
            run_inference_for_lora_stage(
                lora_model_file=vehicle_lora_file,
                prompt=args.prompt,
                output_base_dir=lora_output_root / "lora_only",
                num_videos=args.num_videos,
                nproc_per_node=args.nproc_per_node,
                num_steps=args.num_steps,
                fps=args.fps,
                guidance=args.guidance,
            )
        else:
            print("Error: LoRA checkpoint not found. Aborting 'lora' stage.", file=sys.stderr)
    else:
        print("Skipping LoRA inference.")

    print("\\n" + "=" * 80)
    print("Inference script finished.")
    print(f"All outputs are saved in: {outputs_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
