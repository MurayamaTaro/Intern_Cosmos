"""
Cosmos-Predict1 (diffusion, text-to-video, 7B) の単一データセット学習用スクリプト。
- Cosmosシステムは非常に厳密なため、Hydra風のキー(trainer.*, optimizer.*, dataloader_*.*, model.*)の命名やコマンドライン渡しには注意。
- logs/配下にstdout.log（デバッグ用, checkpoints/配下にできるログと被りあり）、loss_history.csvが生成される。
"""

import argparse
import os
import subprocess
import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime


def parse_loss_from_log(log_file_path: Path) -> pd.DataFrame:
    """ログファイルからイテレーションと損失を抽出して DataFrame として返す。
    """
    print(f"Parsing loss from log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # 損失行を抽出する正規表現
        pattern = re.compile(r"every_n_impl\]\s+(\d+)\s+:.*?Loss:\s+([-\d\.]+)")
        matches = pattern.findall(log_content)

        if not matches:
            print("Warning: No loss values found in the log file. The log format might have changed.", file=sys.stderr)
            return pd.DataFrame(columns=["iteration", "loss"])

        df = pd.DataFrame(matches, columns=["iteration", "loss"]).astype({"iteration": int, "loss": float})
        print(f"Successfully parsed {len(df)} loss entries.")
        return df

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)
        return pd.DataFrame(columns=["iteration", "loss"])
    except Exception as e:
        print(f"An error occurred during log parsing: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["iteration", "loss"])


def run_training(args: argparse.Namespace, run_name: str) -> Path | None:
    """単一データセットでの学習を 1 回実行し、保存された最新 LoRA 重みのパスを返す。"""
    workspace_root = Path.cwd().absolute()

    current_experiment_name = f"text2world_7b_lora_my/{run_name}"

    # ログディレクトリ: stdout.log（デバッグ用）とloss_history.csv はここに並べて保存
    log_dir = workspace_root / "logs" / current_experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "stdout.log"

    # データセットパス: ユーザ指定の絶対パスをそのまま使う
    dataset_path = Path(args.dataset_path).resolve()

    # 自動再開(単一実験内)ロジック
    # 既存チェックポイントがあればそこから再開。なければベースモデルから。
    load_path = None
    checkpoint_dir = (
        workspace_root / "checkpoints" / "posttraining" / "diffusion_text2world"
        / current_experiment_name / "checkpoints"
    )
    if checkpoint_dir.exists():
        manifest_files = [p for p in checkpoint_dir.glob("iter_*.pt") if re.fullmatch(r"iter_\d+\.pt", p.name)]
        if manifest_files:
            latest_manifest = max(manifest_files, key=lambda p: int(re.search(r"iter_(\d+)\.pt", p.name).group(1)))
            potential_load_path = latest_manifest.with_name(f"{latest_manifest.stem}_model.pt")
            if potential_load_path.exists():
                load_path = potential_load_path
                print(f"Found existing checkpoint. Resuming from: {load_path}")
            else:
                print(f"Warning: Manifest found but model weights file missing: {potential_load_path}", file=sys.stderr)

    if load_path is None:
        # 単一実験なので、既存がなければ常にベースモデルから開始
        load_path = workspace_root / "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt"
        print(f"Starting from the base model: {load_path}")

    # latent shape は VAE 8x downsample 前提の計算
    latent_height = args.resolution[0] // 8
    latent_width = args.resolution[1] // 8
    latent_shape = f"[{16},{16},{latent_height},{latent_width}]"

    command = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m", "cosmos_predict1.diffusion.training.train",
        "--config=cosmos_predict1/diffusion/training/config/config.py",
        "--", f"experiment=text2world_7b_lora_my",
        f"job.name={current_experiment_name}",
        f"checkpoint.load_path={load_path}",
        f"trainer.max_iter={args.max_iter}",
        f"trainer.seed={args.seed}",
        f"optimizer.lr={args.learning_rate}",
        f"model.peft_control.rank={args.lora_rank}",
        f"model.peft_control.scale={args.scale}",
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
        f"trainer.grad_accum_iter={args.grad_accum_iter}",
    ]

    # PYTHONPATHにCWD追加
    new_env = os.environ.copy()
    new_env["PYTHONPATH"] = str(workspace_root) + ":" + new_env.get("PYTHONPATH", "")

    print("\n" + "=" * 80)
    print(f"Starting single-dataset training: {current_experiment_name}")
    print(f"Dataset dir: {dataset_path}")
    print(f"Log file: {log_file_path}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)

    # 標準出力をstdout.logに書き出し
    process = subprocess.Popen(
        command,
        cwd=workspace_root,
        env=new_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            log_file.write(line)
    process.wait()

    # 損失のCSVをstdout.logと同じ場所に保存
    loss_df = parse_loss_from_log(log_file_path)
    if not loss_df.empty:
        loss_csv_path = log_dir / "loss_history.csv"
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"Loss history saved to: {loss_csv_path}")

    if process.returncode == 0:
        print("\nTraining completed successfully.")
        # 最新のイテレーションに対応する *_model.pt を返す
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
            return None

        manifest_files = [p for p in checkpoint_dir.glob("iter_*.pt") if re.fullmatch(r"iter_\d+\.pt", p.name)]
        if not manifest_files:
            print(f"Error: No FSDP manifest file (e.g., iter_00000100.pt) found in {checkpoint_dir}", file=sys.stderr)
            return None

        latest_manifest = max(manifest_files, key=lambda p: int(re.search(r"iter_(\d+)\.pt", p.name).group(1)))
        model_weights_file = latest_manifest.with_name(f"{latest_manifest.stem}_model.pt")
        if not model_weights_file.exists():
            print(f"Error: Corresponding model weights file not found: {model_weights_file}", file=sys.stderr)
            return None

        print(f"Latest model weights: {model_weights_file}")
        return model_weights_file
    else:
        print(f"\nTraining failed with return code {process.returncode}.", file=sys.stderr)
        print(f"Check the log for details: {log_file_path}", file=sys.stderr)
        return None


def main():
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Run a single-dataset post-training for Cosmos-Predict1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA matrices.")
    parser.add_argument("--max_iter", type=int, default=8000, help="Total training iterations.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA scaling factor (alpha/rank).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[288, 512],
                        help="Video resolution (height width). Must be multiples of 16.")
    parser.add_argument("--nproc_per_node", type=int, default=8, help="Number of GPUs to use.")
    parser.add_argument("--grad_accum_iter", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="name of the dataset directory.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Absolute path to the dataset directory.")
    args = parser.parse_args()

    # 引数の検証
    if args.resolution[0] % 16 != 0 or args.resolution[1] % 16 != 0:
        print(f"Error: Resolution ({args.resolution[0]}x{args.resolution[1]}) must be multiples of 16.", file=sys.stderr)
        sys.exit(1)

    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path does not exist: {args.dataset_path}", file=sys.stderr)
        sys.exit(1)

    # 実験名の一意化
    run_name = (
        f"{args.dataset_name}_r{args.lora_rank}_iter{args.max_iter}_bs{args.batch_size_per_gpu}"
        f"_accum{args.grad_accum_iter}_scale{args.scale}"
        f"_lr{args.learning_rate:.0e}_seed{args.seed}"
    )
    print(f"Generated Run Name: {run_name}")

    # 単一データセット学習を1回実行
    final_weights = run_training(args=args, run_name=run_name)
    if final_weights is None:
        print("Critical Error: Training failed.", file=sys.stderr)
        sys.exit(1)

    end_time = datetime.now()
    total_duration = end_time - start_time

    print("\n" + "#" * 80)
    print("# Single-dataset post-training finished successfully! #")
    print(f"# Final LoRA weights are at: {final_weights}")
    print(f"# Total execution time: {total_duration}")
    print("#" * 80)


if __name__ == "__main__":
    main()
