import subprocess
import os
import datetime
import sys

def main():
    """
    cosmos-predict1 diffusion 7B text-to-world の追加学習を実行するスクリプト。
    公式のLoRA設定ファイルの構造に合わせた、最小限の必須設定のみで学習。
    """
    # --- 1. 実験設定 ---
    experiment_name = "lora_test_with_nemo_assets"
    base_output_dir = "posttrain_test_v1"

    # LoRAのベースとなる学習済みモデルを指定します。
    pretrained_model_path = "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt"

    # --- 2. 出力ディレクトリの作成 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{experiment_name}_{timestamp}"
    exp_dir = os.path.join(base_output_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"実験結果は次のディレクトリに保存されます: {exp_dir}")

    # --- 3. 学習コマンドの組み立て ---
    train_script = "cosmos_predict1/diffusion/training/train.py"
    config_file = "cosmos_predict1/diffusion/config/config.py"

    # torchrunを使用して8GPUで実行
    command = [
        "torchrun",
        "--nproc_per_node=8",
        train_script,
        f"--config={config_file}",
        "--",
        # --- optsで設定を上書き ---
        # 1. 公式のLoRA設定を読み込む
        "experiment=Cosmos_Predict1_Text2World_7B_Post_trained_lora",

        # 2. VAE/Tokenizer設定 (experiment.pyファイルの構造に合わせて指定)
        # "override /vae=cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",

        # 3. 最小限の必須項目のみ指定
        f"job.name={experiment_name}",
        f"job.path_local={exp_dir}",
        f"checkpoint.load_path={pretrained_model_path}",
        f"checkpoint.lora_weight_path={os.path.join(exp_dir, 'lora_weights.pt')}",

        # 4. テスト用の学習イテレーション設定
        "trainer.max_iter=200",
        "checkpoint.save_iter=100",
    ]

    # --- 4. 実行コマンドの保存 ---
    command_path = os.path.join(exp_dir, "command.sh")
    with open(command_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(" \\\n".join(command))
        f.write("\n")
    print(f"実行コマンドを {command_path} に保存しました。")

    # --- 5. 学習の実行とログ保存 ---
    log_path = os.path.join(exp_dir, "stdout.log")
    print(f"学習を開始します。ログは {log_path} を確認してください。")
    print("-" * 50)

    try:
        with open(log_path, "w") as log_file:
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
                log_file.write(line)

            process.stdout.close()
            return_code = process.wait()

        if return_code != 0:
            print("-" * 50)
            print(f"\nエラーが発生しました。終了コード: {return_code}")
            print(f"詳細はログファイルを確認してください: {log_path}")
        else:
            print("-" * 50)
            print(f"\n学習が正常に終了しました。")
            print(f"LoRA重みやログは {exp_dir} に保存されています。")

    except FileNotFoundError:
        print(f"エラー: 実行ファイルが見つかりません: {train_script} または torchrun")
        print("このスクリプトはワークスペースのルートディレクトリから実行してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
