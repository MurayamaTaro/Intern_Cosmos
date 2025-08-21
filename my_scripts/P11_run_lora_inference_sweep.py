import subprocess
import sys
from pathlib import Path
import itertools

def main():
    # --- 共通設定 ---
    # これらの値は、すべての推論ジョブで共通して使用されます。
    common_args = {
        "num_videos": 1,
        "num_steps": 20,
        "fps": 24,
        # 実行したいステージをリストで指定: 'original', 'vehicle', 'final'
        "stages_to_run": ["vehicle"],
    }

    # --- スイープ対象のパラメータ ---

    # 1. 実験名のリスト
    # P08/P09で学習させた実験の短い名前を指定します。
    experiment_names = [
        "r16_iter500_bs1_accum2_scale2.0_lr5e-05_seed0",
    ]

    # 2. (プロンプト, 推論名) のペアのリスト
    # 推論名は、出力ビデオが保存されるディレクトリ名の一部になります。
    prompts_and_inference_names = [
        # ("A truck on the side of the road next to a house.",
        #  "vehicle_00025"),
        # ("A person is driving a car on a city street, and there are several cars parked on the side of the road.",
        #  "vehicle_00048"),
        # ("There is a car driving on a winding road surrounded by trees and hills.",
        #  "vehicle_00057"),
        # ("There is a red car racing on an asphalt road in front of a large crowd of people who are watching the race.",
        #  "vehicle_00061"),
        ("There is a traffic jam on the road and police cars are directing traffic.",
         "vehicle_00075"),
    ]

    # 3. guidance スケールのリスト
    guidance_values = [5.5]

    # --- スクリプト実行 ---
    script_to_run = Path("/workspace/my_scripts/P10_run_lora_inference.py")

    # experiment_names, prompts_and_inference_names, guidance_values の直積を生成
    job_combinations = list(itertools.product(experiment_names, prompts_and_inference_names, guidance_values))

    print(f"Total number of inference jobs to run: {len(job_combinations)}")

    for i, (exp_name, (prompt, base_inf_name), guidance) in enumerate(job_combinations):
        print("\n" + "="*80)
        print(f"Starting job {i+1}/{len(job_combinations)}")
        print(f"  Experiment Name: {exp_name}")
        print(f"  Guidance       : {guidance}")
        print(f"  Base Inf Name  : {base_inf_name}")
        print(f"  Prompt         : {prompt}")
        print("="*80)

        # guidance 値に基づいて新しい推論名を生成
        inf_name = f"guidance{guidance}/{base_inf_name}"

        command = [
            sys.executable,  # 現在のPythonインタプリタを使用
            str(script_to_run),
            "--experiment_name", exp_name,
            "--prompt", prompt,
            "--inference_name", inf_name,
            "--num_videos", str(common_args["num_videos"]),
            "--num_steps", str(common_args["num_steps"]),
            "--fps", str(common_args["fps"]),
            "--guidance", str(guidance),
            "--stages", *common_args["stages_to_run"], # stages_to_run をコマンドに追加
        ]

        try:
            print(f"Running command: {' '.join(command)}")
            # サブプロセスを実行し、出力をリアルタイムで表示
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)

            process.wait()
            returncode = process.returncode

            if returncode != 0:
                print(f"\n[ERROR] Job {i+1} failed with return code {returncode}.", file=sys.stderr)
            else:
                print(f"\n[SUCCESS] Job {i+1} completed successfully.")

        except Exception as e:
            print(f"An error occurred while running job {i+1}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
