import argparse
import subprocess
import time
import os
import re
import sys

def sanitize_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)[:50]

def main():
    parser = argparse.ArgumentParser(description="Run batch inference for Cosmos-Predict1 models.")
    parser.add_argument("--model_type", type=str, required=True, choices=["text2world", "video2world"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output_tmp")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--guidance", type=float, default=9.0)
    args = parser.parse_args()

    if args.model_type == "video2world" and not args.image_path:
        parser.error("--image_path is required for --model_type video2world")
    if not os.path.exists(args.image_path or "") and args.model_type == "video2world":
        parser.error(f"Image path not found: {args.image_path}")
    if args.num_steps is None:
        args.num_steps = 90 if args.model_type == "video2world" else 35

    # 出力先ディレクトリ
    output_dir = os.path.join("inference", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # infoファイルにまとめて記録
    info_path = os.path.join(output_dir + "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Model type: {args.model_type}\n")
        if args.model_type == "video2world":
            f.write(f"Image path: {args.image_path}\n")
        f.write(f"Num seeds: {args.num_seeds}\n")
        f.write(f"Num steps: {args.num_steps}\n")
        f.write(f"FPS: {args.fps}\n")
        f.write(f"Guidance: {args.guidance}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")

    start_time = time.time()
    for i in range(args.num_seeds):
        seed = args.start_seed + i
        video_name = f"{args.output_dir}_{seed}"
        video_path = os.path.join(output_dir, f"{video_name}.mp4")

        cmd = ["torchrun", "--nproc_per_node", "1"]
        if args.model_type == "text2world":
            cmd += [
                "cosmos_predict1/diffusion/inference/text2world.py",
                "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Text2World",
            ]
        else:
            cmd += [
                "cosmos_predict1/diffusion/inference/video2world.py",
                "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Video2World",
                "--input_image_or_video_path", args.image_path,
                "--guidance", str(args.guidance),
                "--offload_prompt_upsampler",
            ]
        cmd += [
            "--num_gpus", "1",
            "--checkpoint_dir", "checkpoints",
            "--prompt", args.prompt,
            "--num_steps", str(args.num_steps),
            "--video_save_folder", output_dir,
            "--video_save_name", video_name,
            "--seed", str(seed),
            "--fps", str(args.fps),
            "--disable_guardrail",
            "--disable_prompt_upsampler",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error for seed {seed}: {e}")
            continue

    elapsed_time = time.time() - start_time
    print(f"\nBatch inference finished. Total time: {elapsed_time:.2f} seconds.")
    print(f"Results and info: {output_dir}")

if __name__ == "__main__":
    main()
