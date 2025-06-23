import argparse
import os
import subprocess
import sys
import re
from pathlib import Path

def find_checkpoint_path(experiment_path: Path, task_name: str) -> Path | None:
    """指定されたタスクの最新のLoRAチェックポイントパスを見つける。"""
    task_path = experiment_path / task_name
    checkpoint_dir = task_path / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found for task '{task_name}' at {checkpoint_dir}", file=sys.stderr)
        return None

    checkpoints = list(checkpoint_dir.glob("iter_*_model.pt"))
    if not checkpoints:
        print(f"Error: No model checkpoint file found in {checkpoint_dir}", file=sys.stderr)
        return None

    latest_checkpoint = max(
        checkpoints,
        key=lambda p: int(re.search(r"iter_(\d+)_model\.pt", p.name).group(1))
    )
    print(f"Found checkpoint for task '{task_name}': {latest_checkpoint}")
    return latest_checkpoint

def run_inference_for_stage(
    lora_checkpoint_path: Path,
    prompt: str,
    output_base_dir: Path,
    num_videos: int,
    nproc_per_node: int,
    num_steps: int,
    fps: int,
    guidance: float,
):
    """指定されたステージ（LoRA重み）でバッチ推論を実行する。"""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "-" * 80)
    print(f"Running inference for stage: {output_base_dir.name}")
    print(f"Using LoRA checkpoint: {lora_checkpoint_path}")
    print(f"Output directory: {output_base_dir}")
    print("-" * 80)

    for i in range(num_videos):
        seed = i
        seed_output_dir = output_base_dir / f"seed_{seed}"
        seed_output_dir.mkdir(exist_ok=True)

        print(f"  Generating video for seed {seed}...")

        command = [
            "torchrun", f"--nproc_per_node={nproc_per_node}",
            "-m", "cosmos_predict1.diffusion.inference.text2world",
            "--",
            "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Text2World_post-trained-lora",
            "--checkpoint_dir", str(lora_checkpoint_path),
            "--prompt", prompt,
            "--seed", str(seed),
            "--video_save_folder", str(seed_output_dir),
            "--num_steps", str(num_steps),
            "--fps", str(fps),
            "--guidance", str(guidance),
            "--disable_guardrail",
        ]

        try:
            # ★変更点: モデルキャッシュの場所を指定する環境変数を設定
            new_env = os.environ.copy()
            new_env["TRANSFORMERS_CACHE"] = "/workspace/checkpoints"

            process = subprocess.Popen(
                command,
                env=new_env, # ★変更点: カスタム環境変数を渡す
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            generated_videos = list(seed_output_dir.glob("*.mp4"))
            if not generated_videos:
                raise FileNotFoundError("Inference script did not produce an MP4 file.")

            original_video_path = generated_videos[0]
            final_video_path = seed_output_dir / "video.mp4"
            original_video_path.rename(final_video_path)

            prompt_file_path = seed_output_dir / "prompt.txt"
            with open(prompt_file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            print(f"  Successfully generated: {final_video_path}")

        except subprocess.CalledProcessError as e:
            print(f"\n  Error: Inference process failed with return code {e.returncode} for seed {seed}.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\n  An unexpected error occurred for seed {seed}: {e}", file=sys.stderr)
            continue

def main():
    parser = argparse.ArgumentParser(
        description="Run inference using LoRA weights from a continual learning experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True,
        help="The name of the experiment directory, e.g., 'text2world_7b_lora_panda70m_r16_..._seed0'"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The text prompt to use for video generation."
    )
    parser.add_argument(
        "--num_videos", type=int, default=5,
        help="Number of videos to generate for each stage (using different seeds)."
    )
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs for inference. 1 is recommended.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second of the generated video.")
    parser.add_argument("--guidance", type=float, default=8.0, help="Guidance scale.")

    args = parser.parse_args()

    workspace_root = Path("/workspace")
    checkpoints_root = workspace_root / "checkpoints/posttraining/diffusion_text2world"
    experiment_path = checkpoints_root / args.experiment_name

    if not experiment_path.exists():
        print(f"Error: Experiment directory not found at {experiment_path}", file=sys.stderr)
        sys.exit(1)

    tasks = ["vehicle", "cooking", "sports"]
    vehicle_lora_path = find_checkpoint_path(experiment_path, tasks[0])
    final_lora_path = find_checkpoint_path(experiment_path, tasks[-1])

    if not vehicle_lora_path or not final_lora_path:
        print("Could not find all required checkpoints. Aborting.", file=sys.stderr)
        sys.exit(1)

    output_root = workspace_root / "lora_inference" / args.experiment_name

    # Stage 1: vehicle_only
    run_inference_for_stage(
        lora_checkpoint_path=vehicle_lora_path,
        prompt=args.prompt,
        output_base_dir=output_root / "vehicle_only",
        num_videos=args.num_videos,
        nproc_per_node=args.nproc_per_node,
        num_steps=args.num_steps,
        fps=args.fps,
        guidance=args.guidance,
    )

    # Stage 2: final
    run_inference_for_stage(
        lora_checkpoint_path=final_lora_path,
        prompt=args.prompt,
        output_base_dir=output_root / "final",
        num_videos=args.num_videos,
        nproc_per_node=args.nproc_per_node,
        num_steps=args.num_steps,
        fps=args.fps,
        guidance=args.guidance,
    )

    print("\n" + "="*80)
    print("Inference script finished.")
    print(f"All outputs are saved in: {output_root}")
    print("="*80)

if __name__ == "__main__":
    main()
