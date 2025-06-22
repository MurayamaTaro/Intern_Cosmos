import argparse
import os
import subprocess
import sys
from pathlib import Path
import re

def run_training(
    experiment_name: str,
    task_name: str,
    nproc_per_node: int,
    previous_lora_path: Path | None,
) -> Path | None:
    """
    指定された実験設定でトレーニングを実行し、生成されたLoRAチェックポイントのパスを返す。

    Args:
        experiment_name (str): 実行する実験のベース名。
        task_name (str): 現在のタスク名 (e.g., "vehicle")。
        nproc_per_node (int): 使用するGPUの数。
        output_root (str): チェックポイントとログのルートディレクトリ。
        previous_lora_path (Path | None): 前のタスクで生成されたLoRAのパス。Noneの場合はベースモデルから開始。

    Returns:
        Path | None: 正常に完了した場合、最後に保存されたLoRAチェックポイントのパス。失敗した場合はNone。
    """
    workspace_root = Path.cwd().absolute()

    # フレームワークの命名規則に合わせて、出力ディレクトリのパスを正確に構築する
    current_experiment_name = f"{experiment_name}_{task_name}"
    # フレームワークは 'posttraining/diffusion_text2world' というパスを生成する
    exp_dir = workspace_root / "posttraining" / "diffusion_text2world" / current_experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # シンボリックリンクを作成
    symlink_path = exp_dir / "cosmos_predict1"
    source_path = workspace_root / "cosmos_predict1"
    if not symlink_path.exists():
        os.symlink(source_path, symlink_path, target_is_directory=True)

    log_file_path = exp_dir / "stdout.log"

    # データセットのパスを絶対パスで構築
    dataset_path = workspace_root / "datasets/posttrain_panda70m" / task_name

    # --- ここからが継続学習のコアロジック ---
    if previous_lora_path:
        # 2番目以降のタスク：前のタスクのLoRAを読み込む
        load_path = previous_lora_path
        print(f"Info: Continuing training from previous LoRA: {load_path}")
    else:
        # 最初のタスク：ベースモデルを読み込む
        load_path = workspace_root / "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt"
        print(f"Info: Starting training from base model: {load_path}")
    # --- ここまでが継続学習のコアロジック ---

    command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "-m", "cosmos_predict1.diffusion.training.train",
        "--config=cosmos_predict1/diffusion/training/config/config.py",
        "--", f"experiment={experiment_name}", # ベースとなる実験設定名は固定
        f"checkpoint.load_path={load_path}",
        # データセットのパスをコマンドラインから上書き
        f"dataloader_train.dataset.dataset_dir={dataset_path}",
        f"dataloader_val.dataset.dataset_dir={dataset_path}",
        # ログが見やすいように、ジョブ名もタスクごとに変更
        f"job.name={current_experiment_name}",
    ]

    new_env = os.environ.copy()
    new_env["PYTHONPATH"] = str(workspace_root) + ":" + new_env.get("PYTHONPATH", "")

    print("\n" + "=" * 80)
    print(f"Starting task: {task_name} ({current_experiment_name})")
    print(f"Output directory (CWD): {exp_dir}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)

    try:
        process = subprocess.Popen(
            command, cwd=exp_dir, env=new_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1
        )
        with open(log_file_path, 'w') as log_file:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                log_file.write(line)
        process.wait()

        if process.returncode == 0:
            print(f"\nTask '{task_name}' completed successfully.")
            # 正しいチェックポイントディレクトリを探す
            checkpoint_dir = exp_dir / "checkpoints"
            # `iter_..._model.pt` という形式のファイルを探す
            checkpoints = list(checkpoint_dir.glob("iter_*_model.pt"))
            if not checkpoints:
                print(f"Error: No model checkpoint file found in {checkpoint_dir}", file=sys.stderr)
                return None

            # イテレーション番号が最大のファイルを見つける
            latest_checkpoint = max(
                checkpoints,
                key=lambda p: int(re.search(r"iter_(\d+)_model\.pt", p.name).group(1))
            )
            print(f"Saved checkpoint for next task: {latest_checkpoint}")
            return latest_checkpoint
        else:
            print(f"\nTask '{task_name}' failed with return code {process.returncode}.", file=sys.stderr)
            print(f"Check the log for details: {log_file_path}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"An unexpected error occurred during task {task_name}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run a continual learning loop for Cosmos-Predict1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 引数はP06から流用・調整
    parser.add_argument(
        "--experiment_name", type=str,
        default="text2world_7b_lora_panda70m_vehicle_test",
        help="The base name of the experiment defined in experiment.py."
    )
    parser.add_argument("--nproc_per_node", type=int, default=8, help="Number of GPUs to use.")
    args = parser.parse_args()

    # 継続学習するタスクのリスト
    tasks = ["vehicle", "cooking", "sports"]

    previous_lora_path = None
    for i, task_name in enumerate(tasks):
        print("\n" + "#" * 80)
        print(f"# Starting Continual Learning Stage {i+1}/{len(tasks)}: Task '{task_name}'")
        print("#" * 80)

        previous_lora_path = run_training(
            experiment_name=args.experiment_name,
            task_name=task_name,
            nproc_per_node=args.nproc_per_node,
            previous_lora_path=previous_lora_path,
        )

        if previous_lora_path is None:
            print(f"\nCritical Error: Stage {i+1} ('{task_name}') failed. Aborting continual learning loop.")
            # 最初のタスクが成功している可能性があるので、ここで終了する
            sys.exit(1)

    if previous_lora_path:
        print("\n" + "#" * 80)
        print("# Continual learning loop finished successfully! #")
        print(f"# Final LoRA weights are located at: {previous_lora_path}")
        print("#" * 80)


if __name__ == "__main__":
    main()
