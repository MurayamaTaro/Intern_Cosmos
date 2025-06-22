import subprocess
import itertools
from pathlib import Path
import sys

def main():
    # --- 探索するハイパーパラメータの範囲を定義 ---
    param_grid = {
        'lora_rank': [8, 16], # 8, 16
        'learning_rate': [5e-5, 1e-4],
        'max_iter': [3000], # 3000
        'seed': [0],
    }

    # パラメータの組み合わせを生成
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total number of experiments to run: {len(param_combinations)}")

    # 既存のスクリプトのパス
    script_to_run = '/workspace/my_scripts/P08_run_continual_learning.py'

    for i, params in enumerate(param_combinations):
        print("\n" + "="*80)
        print(f"Starting experiment {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        print("="*80)

        # 実行するコマンドを構築
        command = [
            sys.executable,  # 現在のPythonインタプリタを使用
            script_to_run,
            '--lora_rank', str(params['lora_rank']),
            '--learning_rate', str(params['learning_rate']),
            '--max_iter', str(params['max_iter']),
            '--seed', str(params['seed']),
            # --- その他の固定パラメータ ---
            # '--batch_size_per_gpu', '1',
            # '--nproc_per_node', '8',
        ]

        # 実行
        try:
            # ログをファイルに保存しつつ、コンソールにも表示
            run_name = (
                f"r{params['lora_rank']}_lr{params['learning_rate']}"
                f"_iter{params['max_iter']}_seed{params['seed']}"
            )
            log_dir = Path(f"/workspace/sweep_logs/{run_name}")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / "sweep_stdout.log"

            print(f"Running command: {' '.join(command)}")
            print(f"Log file will be saved to: {log_file_path}")

            # subprocess.runを使って同期的に実行
            result = subprocess.run(
                command,
                capture_output=True, # stdoutとstderrをキャプチャ
                text=True,
                encoding='utf-8'
            )

            # ログをファイルに書き込む
            with open(log_file_path, 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)

            # コンソールにも出力（エラーがあった場合などに見やすいように）
            print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)


            if result.returncode != 0:
                print(f"\n[ERROR] Experiment {i+1} failed with return code {result.returncode}.")
                print(f"Check log for details: {log_file_path}")
            else:
                print(f"\n[SUCCESS] Experiment {i+1} completed successfully.")

        except Exception as e:
            print(f"An error occurred while running experiment {i+1}: {e}")

if __name__ == "__main__":
    main()
