import subprocess
from pathlib import Path
import sys

def main():
    # --- 実行したい実験の組み合わせをここに定義 ---
    # 'tasks' キーを追加して、実行したいタスクを指定
    experiment_configs = [
        # vehicle タスクのみ実行
        # {'lora_rank': 8, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 2, 'learning_rate': 2e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # {'lora_rank': 8, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 2, 'learning_rate': 5e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # {'lora_rank': 8, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 4, 'scale': 2, 'learning_rate': 2e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # {'lora_rank': 8, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 4, 'scale': 2, 'learning_rate': 5e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # {'lora_rank': 16, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 2, 'learning_rate': 2e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # {'lora_rank': 16, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 4, 'scale': 2, 'learning_rate': 2e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter

        {'lora_rank': 16, 'max_iter': 500, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 2, 'learning_rate': 5e-5, 'seed': 0, 'tasks': ['vehicle']}, # 1epoch=5000/(16or32)iter=312or624iter
        # 済 {'lora_rank': 16, 'max_iter': 4000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 4, 'scale': 2, 'learning_rate': 5e-5, 'seed': 0, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # 済 {'lora_rank': 16, 'max_iter': 6000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 1, 'learning_rate': 5e-5, 'seed': 1, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
        # 済予定 {'lora_rank': 16, 'max_iter': 6000, 'batch_size_per_gpu': 1, 'grad_accum_iter': 2, 'scale': 2, 'learning_rate': 5e-5, 'seed': 1, 'tasks': ['vehicle', 'cooking', 'sports']}, # 1epoch=5000/(16or32)iter=312or624iter
    ]

    print(f"Total number of experiments to run: {len(experiment_configs)}")

    # 既存のスクリプトのパス
    script_to_run = '/workspace/my_scripts/P08_run_continual_learning.py'

    for i, params in enumerate(experiment_configs):
        print("\n" + "="*80)
        print(f"Starting experiment {i+1}/{len(experiment_configs)}")
        print(f"Parameters: {params}")
        print("="*80)

        # 実行するコマンドを構築
        command = [
            sys.executable,  # 現在のPythonインタプリタを使用
            script_to_run,
            '--lora_rank', str(params['lora_rank']),
            '--max_iter', str(params['max_iter']),
            '--batch_size_per_gpu', str(params['batch_size_per_gpu']),
            '--grad_accum_iter', str(params['grad_accum_iter']),
            '--scale', str(params['scale']),
            '--learning_rate', str(params['learning_rate']),
            '--seed', str(params['seed']),
        ]

        # 'tasks' パラメータが存在し、空でない場合にコマンドに追加
        if 'tasks' in params and params['tasks']:
            command.append('--tasks')
            command.extend(params['tasks'])

        # 実行
        try:
            # ログをファイルに保存しつつ、コンソールにも表示
            # run_name を batch_size_per_gpu を使うように変更
            run_name = (
                f"r{params['lora_rank']}_iter{params['max_iter']}_bs{params['batch_size_per_gpu']}"
                f"_accum{params['grad_accum_iter']}_scale{params['scale']}"
                f"_lr{params['learning_rate']}_seed{params['seed']}"
            )
            log_dir = Path(f"/workspace/sweep_logs/{run_name}")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / "sweep_stdout.log"

            print(f"Running command: {' '.join(command)}")
            print(f"Log file will be saved to: {log_file_path}")

            # subprocess.run の代わりに Popen を使い、出力を直接ファイルにリダイレクト
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT, # 標準エラー出力を標準出力にまとめる
                    text=True,
                    encoding='utf-8'
                )
                # プロセスの終了を待つ
                process.wait()
                returncode = process.returncode

            # 終了コードで成功/失敗を判定
            if returncode != 0:
                print(f"\n[ERROR] Experiment {i+1} failed with return code {returncode}.")
                print(f"Check log for details: {log_file_path}")
                # エラーログをコンソールに表示して確認しやすくする
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    print("--- LOGS ---")
                    print(f.read())
            else:
                print(f"\n[SUCCESS] Experiment {i+1} completed successfully.")

        except Exception as e:
            print(f"An error occurred while running experiment {i+1}: {e}")

if __name__ == "__main__":
    main()
