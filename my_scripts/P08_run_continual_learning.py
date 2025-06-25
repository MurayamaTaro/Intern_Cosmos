import argparse
import os
import subprocess
import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_loss_from_log(log_file_path: Path) -> pd.DataFrame:
    """ログファイルからイテレーションと損失を抽出してDataFrameとして返す。"""
    print(f"Parsing loss from log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # ★変更点: IterSpeedコールバックのログ形式に合わせた正規表現に変更
        # 例: ...iter_speed.py:80:every_n_impl] 20 : ... | Loss: -0.0437
        pattern = re.compile(r"every_n_impl\]\s+(\d+)\s+:.*?Loss:\s+([-\d\.]+)")
        matches = pattern.findall(log_content)

        if not matches:
            print("Warning: No loss values found in the log file. The log format might have changed.", file=sys.stderr)
            return pd.DataFrame(columns=["iteration", "loss"])

        df = pd.DataFrame(matches, columns=["iteration", "loss"])
        df = df.astype({"iteration": int, "loss": float})
        print(f"Successfully parsed {len(df)} loss entries.")
        return df

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)
        return pd.DataFrame(columns=["iteration", "loss"])
    except Exception as e:
        print(f"An error occurred during log parsing: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["iteration", "loss"])


def run_training(
    args: argparse.Namespace,
    task_name: str,
    run_name: str,
    previous_lora_path: Path | None,
) -> Path | None:
    """指定された実験設定でトレーニングを実行し、生成されたLoRAチェックポイントのパスを返す。"""
    workspace_root = Path.cwd().absolute()
    parent_experiment_name = f"{args.experiment_base_name}_{run_name}"
    current_experiment_name = f"{parent_experiment_name}/{task_name}"

    log_dir = workspace_root / "posttraining_logs" / current_experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "stdout.log"

    dataset_path = workspace_root / "datasets/posttrain_panda70m" / task_name

    # 継続的学習のステージに応じて、読み込むチェックポイントのパスを動的に決定する
    if previous_lora_path:
        # 2番目以降のタスクでは、前のタスクの出力チェックポイントを読み込む
        load_path = previous_lora_path
        print(f"Continual learning step. Loading previous checkpoint from: {load_path}")
    else:
        # 最初のタスクでは、ベースモデルを読み込む
        load_path = workspace_root / "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt"
        print(f"First training step. Starting from the base model: {load_path}")

    latent_height = args.resolution[0] // 8
    latent_width = args.resolution[1] // 8
    latent_shape = f"[{16},{16},{latent_height},{latent_width}]"

    command = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m", "cosmos_predict1.diffusion.training.train",
        "--config=cosmos_predict1/diffusion/training/config/config.py",
        "--", f"experiment={args.experiment_base_name}",
        f"job.name={current_experiment_name}",
        # 動的に決定した `load_path` を使用
        f"checkpoint.load_path={load_path}",
        f"trainer.max_iter={args.max_iter}",
        f"trainer.seed={args.seed}",
        f"optimizer.lr={args.learning_rate}",
        f"model.peft_control.rank={args.lora_rank}",
        f"model.latent_shape={latent_shape}",
        f"dataloader_train.dataset.video_size={args.resolution}",
        f"dataloader_train.sampler.dataset.video_size={args.resolution}",
        f"dataloader_val.dataset.video_size={args.resolution}",
        f"dataloader_val.sampler.dataset.video_size={args.resolution}",
        f"dataloader_train.dataset.dataset_dir={dataset_path}",
        f"dataloader_train.sampler.dataset.dataset_dir={dataset_path}",
        f"dataloader_val.dataset.dataset_dir={dataset_path}",
        f"dataloader_val.sampler.dataset.dataset_dir={dataset_path}",
        f"dataloader_train.batch_size={args.batch_size_per_gpu}",
        f"dataloader_val.batch_size={args.batch_size_per_gpu}",
    ]

    new_env = os.environ.copy()
    new_env["PYTHONPATH"] = str(workspace_root) + ":" + new_env.get("PYTHONPATH", "")

    print("\n" + "=" * 80)
    print(f"Starting task: {task_name} ({current_experiment_name})")
    print(f"Log file: {log_file_path}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)

    process = subprocess.Popen(
        command, cwd=workspace_root, env=new_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1
    )
    with open(log_file_path, 'w') as log_file:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            log_file.write(line)
    process.wait()

    loss_df = parse_loss_from_log(log_file_path)
    if not loss_df.empty:
        loss_csv_path = log_dir / "loss_history.csv"
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"Loss history saved to: {loss_csv_path}")

    if process.returncode == 0:
        print(f"\nTask '{task_name}' completed successfully.")
        checkpoint_dir = (
            workspace_root / "checkpoints" / "posttraining" / "diffusion_text2world"
            / current_experiment_name / "checkpoints"
        )
        checkpoints = list(checkpoint_dir.glob("iter_*_model.pt"))
        if not checkpoints:
            print(f"Error: No model checkpoint file found in {checkpoint_dir}", file=sys.stderr)
            return None
        latest_checkpoint = max(
            checkpoints, key=lambda p: int(re.search(r"iter_(\d+)_model\.pt", p.name).group(1))
        )
        print(f"Saved checkpoint for next task: {latest_checkpoint}")
        return latest_checkpoint
    else:
        print(f"\nTask '{task_name}' failed with return code {process.returncode}.", file=sys.stderr)
        print(f"Check the log for details: {log_file_path}", file=sys.stderr)
        return None

def main():
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Run a flexible continual learning loop for Cosmos-Predict1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA matrices.")
    parser.add_argument("--max_iter", type=int, default=8000, help="Total training iterations per task.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[352, 640], help="Video resolution (height width). Must be multiples of 16.")
    parser.add_argument("--experiment_base_name", type=str, default="text2world_7b_lora_panda70m", help="The base name of the experiment in experiment.py.")
    parser.add_argument("--nproc_per_node", type=int, default=8, help="Number of GPUs to use.")
    # ★変更点: --tasks 引数を追加
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["vehicle", "cooking", "sports"],
        help="A list of task names to run in sequence."
    )
    args = parser.parse_args()

    if args.resolution[0] % 16 != 0 or args.resolution[1] % 16 != 0:
        print(f"Error: Resolution ({args.resolution[0]}x{args.resolution[1]}) must be multiples of 16.", file=sys.stderr)
        sys.exit(1)

    global_batch_size = args.batch_size_per_gpu * args.nproc_per_node
    run_name = (
        f"r{args.lora_rank}_iter{args.max_iter}_bs{global_batch_size}"
        f"_lr{args.learning_rate}_seed{args.seed}"
    )
    print(f"Generated Run Name: {run_name}")

    # ★変更点: ハードコードされたリストの代わりに引数を使用
    tasks = args.tasks
    print(f"Tasks to be executed: {tasks}")

    previous_lora_path = None
    for i, task_name in enumerate(tasks):
        print("\n" + "#" * 80)
        print(f"# Starting Continual Learning Stage {i+1}/{len(tasks)}: Task '{task_name}'")
        print("#" * 80)

        previous_lora_path = run_training(
            args=args,
            task_name=task_name,
            run_name=run_name,
            previous_lora_path=previous_lora_path,
        )

        if previous_lora_path is None:
            print(f"\nCritical Error: Stage {i+1} ('{task_name}') failed. Aborting loop.")
            sys.exit(1)

    end_time = datetime.now()
    total_duration = end_time - start_time

    print("\n" + "#" * 80)
    print("# Continual learning loop finished successfully! #")
    print(f"# Final LoRA weights are at: {previous_lora_path}")
    print(f"# Total execution time: {total_duration}")
    print("#" * 80)

if __name__ == "__main__":
    main()
