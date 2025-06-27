import argparse
import os
import subprocess
import sys
import re
import shutil
from pathlib import Path
import torch

def find_lora_checkpoint_file(experiment_path: Path, task_name: str) -> Path | None:
    """指定されたタスクの最新のLoRAチェックポイントファイルパスを見つける。"""
    task_path = experiment_path / task_name
    checkpoint_dir = task_path / "checkpoints"

    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found for task '{task_name}' at {checkpoint_dir}", file=sys.stderr)
        return None

    latest_txt_path = checkpoint_dir / "latest_checkpoint.txt"
    if not latest_txt_path.exists():
        print(f"Error: 'latest_checkpoint.txt' not found in {checkpoint_dir}.", file=sys.stderr)
        return None

    with open(latest_txt_path, 'r') as f:
        latest_iter_name = f.read().strip()

    base_name = latest_iter_name.removesuffix('.pt')
    lora_model_file = checkpoint_dir / f"{base_name}_model.pt"

    if not lora_model_file.exists():
        print(f"Error: Latest checkpoint file '{lora_model_file}' does not exist.", file=sys.stderr)
        return None

    print(f"Found LoRA checkpoint file for task '{task_name}': {lora_model_file}")
    return lora_model_file

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
    """LoRA重みを使ってバッチ推論を実行する。"""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Running inference for LoRA stage: {output_base_dir.name}")
    print(f"Using LoRA file: {lora_model_file}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    tmp_checkpoint_dir = output_base_dir.parent / f"tmp_{output_base_dir.name}"
    try:
        if tmp_checkpoint_dir.exists():
            shutil.rmtree(tmp_checkpoint_dir)
        tmp_checkpoint_dir.mkdir(parents=True)

        print("Preparing a temporary directory with symbolic links...")
        workspace_root = Path("/workspace")
        original_checkpoints_root = workspace_root / "checkpoints"

        required_components = [
            "Cosmos-Tokenize1-CV8x8x8-720p",
            "google-t5",
        ]

        for component_name in required_components:
            source_path = original_checkpoints_root / component_name
            dest_path = tmp_checkpoint_dir / component_name
            if not source_path.exists():
                print(f"Warning: Required component '{source_path}' not found. Skipping.", file=sys.stderr)
                continue
            os.symlink(source_path, dest_path)
            print(f"Created symlink for '{component_name}'.")

        model_type_subdir_name = "Cosmos-Predict1-7B-Text2World_post-trained-lora"
        nested_model_dir = tmp_checkpoint_dir / model_type_subdir_name
        nested_model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading original LoRA checkpoint from: {lora_model_file}")
        original_checkpoint = torch.load(lora_model_file, map_location="cpu")

        lora_state_dict = original_checkpoint['model']
        print("Successfully extracted LoRA weights from the 'model' key.")

        compatible_checkpoint = {
            "model": lora_state_dict,
            "ema": {}
        }
        tmp_model_path = nested_model_dir / "model.pt"
        torch.save(compatible_checkpoint, tmp_model_path)
        print(f"Prepared compatible LoRA checkpoint at: {tmp_model_path}")

        for i in range(num_videos):
            seed = i
            seed_output_dir = output_base_dir / f"seed_{seed}"
            seed_output_dir.mkdir(exist_ok=True)

            print(f"  Generating video for seed {seed}...")

            command = [
                "torchrun", f"--nproc_per_node={nproc_per_node}",
                "-m", "cosmos_predict1.diffusion.inference.text2world",
                "--",
                "--diffusion_transformer_dir", model_type_subdir_name,
                "--checkpoint_dir", str(tmp_checkpoint_dir),
                "--prompt", prompt,
                "--seed", str(seed),
                "--video_save_folder", str(seed_output_dir),
                "--num_steps", str(num_steps),
                "--fps", str(fps),
                "--guidance", str(guidance),
                "--disable_guardrail",
                "--disable_prompt_upsampler"
            ]

            new_env = os.environ.copy()
            new_env["TRANSFORMERS_CACHE"] = str(original_checkpoints_root)

            process = subprocess.Popen(
                command,
                env=new_env,
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
        i = locals().get('i', 'N/A')
        print(f"\n  Error: Inference process failed with return code {e.returncode} for seed {i}.", file=sys.stderr)
    except Exception as e:
        print(f"\n  An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if 'tmp_checkpoint_dir' in locals() and tmp_checkpoint_dir.exists():
            print(f"Cleaning up temporary directory: {tmp_checkpoint_dir}")
            shutil.rmtree(tmp_checkpoint_dir)

def run_inference_for_base_model(
    prompt: str,
    output_base_dir: Path,
    num_videos: int,
    nproc_per_node: int,
    num_steps: int,
    fps: int,
    guidance: float,
):
    """ベースモデルでバッチ推論を実行する。"""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Running inference for base model stage: {output_base_dir.name}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    workspace_root = Path("/workspace")
    original_checkpoints_root = workspace_root / "checkpoints"

    for i in range(num_videos):
        seed = i
        seed_output_dir = output_base_dir / f"seed_{seed}"
        seed_output_dir.mkdir(exist_ok=True)

        print(f"  Generating video for seed {seed}...")

        command = [
            "torchrun", f"--nproc_per_node={nproc_per_node}",
            "-m", "cosmos_predict1.diffusion.inference.text2world",
            "--",
            "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Text2World",
            "--checkpoint_dir", str(original_checkpoints_root),
            "--prompt", prompt,
            "--seed", str(seed),
            "--video_save_folder", str(seed_output_dir),
            "--num_steps", str(num_steps),
            "--fps", str(fps),
            "--guidance", str(guidance),
            "--disable_guardrail",
            "--disable_prompt_upsampler"
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
        "--inference_name", type=str, required=True,
        help="A custom name for this inference run, used as a parent directory for the output."
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True,
        help="The short name of the experiment directory, e.g., 'r8_iter3000_bs8_lr0.0001_seed0'"
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
    parser.add_argument(
        "--skip_base", action="store_true",
        help="Skip inference for the base model stage."
    )

    args = parser.parse_args()

    workspace_root = Path("/workspace")

    # --- Base Model Inference (with skip check) ---
    if not args.skip_base:
        base_output_dir = workspace_root / "lora_inference" / args.inference_name / "original"
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
        print("Skipping base model inference as '--skip_base' flag is set.")

    # --- LoRA Model Inference ---
    checkpoints_root = workspace_root / "checkpoints/posttraining/diffusion_text2world"
    experiment_path = checkpoints_root / f"text2world_7b_lora_panda70m_{args.experiment_name}"

    if not experiment_path.exists():
        print(f"Error: Experiment directory not found at {experiment_path}", file=sys.stderr)
        sys.exit(1)

    tasks = ["vehicle", "cooking", "sports"]

    vehicle_lora_file = find_lora_checkpoint_file(experiment_path, tasks[0])
    final_lora_file = find_lora_checkpoint_file(experiment_path, tasks[-1])

    if not vehicle_lora_file or not final_lora_file:
        print("Could not find all required checkpoints. Aborting.", file=sys.stderr)
        sys.exit(1)

    lora_output_root = workspace_root / "lora_inference" / args.inference_name / args.experiment_name

    # Stage 1: vehicle_only (LoRA)
    run_inference_for_lora_stage(
        lora_model_file=vehicle_lora_file,
        prompt=args.prompt,
        output_base_dir=lora_output_root / "vehicle_only",
        num_videos=args.num_videos,
        nproc_per_node=args.nproc_per_node,
        num_steps=args.num_steps,
        fps=args.fps,
        guidance=args.guidance,
    )

    # Stage 2: final (LoRA)
    run_inference_for_lora_stage(
        lora_model_file=final_lora_file,
        prompt=args.prompt,
        output_base_dir=lora_output_root / "final",
        num_videos=args.num_videos,
        nproc_per_node=args.nproc_per_node,
        num_steps=args.num_steps,
        fps=args.fps,
        guidance=args.guidance,
    )

    print("\n" + "="*80)
    print("Inference script finished.")
    print(f"All outputs are saved in: {workspace_root / 'lora_inference' / args.inference_name}")
    print("="*80)

if __name__ == "__main__":
    main()
