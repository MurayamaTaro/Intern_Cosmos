import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_training(experiment_name: str, nproc_per_node: int, output_root: str):
    """
    指定された実験設定でトレーニングを実行し、出力をログファイルに保存する。
    作業ディレクトリとPYTHONPATH、そしてシンボリックリンクを設定することで、
    出力先と各種パスの問題を解決する。

    Args:
        experiment_name (str): 実行する実験の名前。
        nproc_per_node (int): 使用するGPUの数。
        output_root (str): チェックポイントとログのルートディレクトリ。
    """
    workspace_root = Path.cwd().absolute()
    exp_dir = workspace_root / output_root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    symlink_path = exp_dir / "cosmos_predict1"
    source_path = workspace_root / "cosmos_predict1"
    if not symlink_path.exists():
        os.symlink(source_path, symlink_path, target_is_directory=True)
        print(f"Info: Created symlink for robust path handling: {symlink_path} -> {source_path}")

    log_file_path = exp_dir / "stdout.log"

    # --- ここからが重要な修正 ---
    # すべてのハードコードされた相対パスを絶対パスに変換する
    base_model_path = workspace_root / "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt"
    vae_checkpoint_dir = workspace_root / "checkpoints/Cosmos-Tokenize1-CV8x8x8-720p"
    vae_encoder_path = vae_checkpoint_dir / "encoder.jit"
    vae_decoder_path = vae_checkpoint_dir / "decoder.jit"
    vae_mean_std_path = vae_checkpoint_dir / "mean_std.pt"
    # --- ここまでが重要な修正 ---

    command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "-m", "cosmos_predict1.diffusion.training.train",
        "--config=cosmos_predict1/diffusion/training/config/config.py",
        "--", f"experiment={experiment_name}",
        # すべてのパスをコマンドラインから絶対パスで上書きする
        f"checkpoint.load_path={base_model_path}",
    ]

    new_env = os.environ.copy()
    new_env["PYTHONPATH"] = str(workspace_root) + ":" + new_env.get("PYTHONPATH", "")

    print("=" * 80)
    print(f"Starting experiment: {experiment_name}")
    print(f"Output directory (CWD): {exp_dir}")
    print(f"PYTHONPATH for subprocess: {new_env['PYTHONPATH']}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)

    try:
        process = subprocess.Popen(
            command,
            cwd=exp_dir,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        with open(log_file_path, 'w') as log_file:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                log_file.write(line)

        process.wait()

        if process.returncode == 0:
            print("\n" + "=" * 80)
            print(f"Experiment '{experiment_name}' completed successfully.")
            print(f"Log saved to: {log_file_path}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80, file=sys.stderr)
            print(f"Experiment '{experiment_name}' failed with return code {process.returncode}.", file=sys.stderr)
            print(f"Check the log for details: {log_file_path}", file=sys.stderr)
            print("=" * 80, file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: 'torchrun' command not found. Make sure you are in the correct environment.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single training task for Cosmos-Predict1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="text2world_7b_lora_panda70m_vehicle_test",
        help="The name of the experiment defined in experiment.py."
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="Number of GPUs to use."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="posttrain",
        help="The root directory for saving experiment outputs (logs, checkpoints)."
    )
    args = parser.parse_args()

    run_training(
        experiment_name=args.experiment_name,
        nproc_per_node=args.nproc_per_node,
        output_root=args.output_root
    )


if __name__ == "__main__":
    main()
