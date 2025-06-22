# Cosmos

## 環境構築
- OS: Amazon Linux 2023
- EC2インスタンスはオンプレではなくAWSのクラウド上に立っているのでプロキシ設定は不要

## S3
- aws s3 ls --profile murayama
- aws s3 cp ./clips/ s3://dnjp-riron-murayama/clips/ --recursive

## git設定
- git --version
- sudo dnf install -y git
- クローン
  - git clone https://github.com/nvidia-cosmos/cosmos-predict1.git
  - mv cosmos-predict1 Cosmos
  - cd Cosmos
- リモート追加
  - git remote remove origin
  - Github/Gitlabでリモート作成
  - git remote add origin 作成したURL（例：https://github.com/MurayamaTaro/Cosmos.git）
  - git remote -v
- 実験用ブランチを切る（新ブランチを作成し切替）
  - git checkout -b expr-1 main
  - git branchで確認
- コードを改造してコミット
- 実験ブランチをorigin/実験ブランチにpush
  - git push -u origin expr-1
- 実験ブランチをmainにマージ
  - git checkout main
  - git merge expr-1
- リモートにpush
  - git push origin main

## Docker
- dockerインストール
  - sudo yum update -y
  - sudo yum install -y docker
- Dockerサービス起動＆自動起動設定
  - sudo systemctl start docker
  - sudo systemctl enable docker
- ec2-userをdockerグループに追加
  - sudo usermod -aG docker ec2-user
  - 再起動
- NVIDIA Docker（nvidia-docker2）のインストール（Amazom linux 2向け）
  - 必要なリポジトリ追加
    - distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    - curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    - sudo yum update -y
    - sudo yum install -y nvidia-docker2
    - sudo systemctl restart docker
  - GPU情報が表示されるか確認
    - docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi


## Dockerコマンドメモ
- docker ps -a
- docker stop (ID)
- docker rm (ID)


## 推論
- モデルパラメータをダウンロード
  - PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World
  - PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Video2World
- テキスト条件付け
  - torchrun --nproc_per_node 8 \
  cosmos_predict1/diffusion/inference/text2world.py \
  --num_gpus 8 \
  --checkpoint_dir checkpoints \
  --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
  --prompt "$PROMPT" \
  --num_steps 35 \
  --video_save_folder outputs \
  --video_save_name splash_boy \
  --seed 0 \
  --fps 24 \
  --disable_guardrail \
  --disable_prompt_upsampler
- テキスト＋画像条件付け
  - torchrun --nproc_per_node 4 \
    cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus 4 \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --prompt "${PROMPT}" \
    --num_steps 90 \
    --input_image_or_video_path "denmaru/cut2.jpg" \
    --video_save_folder denmaru \
    --video_save_name output0 \
    --seed 0 \
    --fps 12 \
    --guidance 9 \
    --disable_guardrail \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler

## 学習
- LoRA学習
  - デフォルトではmax_iter=1000
  - export OUTPUT_ROOT=checkpoints
  - torchrun --nproc_per_node=8 \
    -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_lora_example_cosmos_nemo_assets
- 比較（共通のアップサンプル後プロンプトを使う必要があることに注意）
  - (1) Base モデルでアップサンプル→動画＆txt 出力
  torchrun --nproc_per_node=8 cosmos_predict1/diffusion/inference/text2world.py \
    --num_gpus 8 \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "A video of sks teal robot picking up a green cube and placing it on a red platform." \
    --num_steps 35 \
    --video_save_folder outputs/comparison/prompt1/before \
    --video_save_name before \
    --disable_guardrail
  - 生成されたアップサンプル済みプロンプトを読み込む
  UPSAMPLED_PROMPT=$(< outputs/comparison/prompt1/before/before.txt)
  - (2) LoRA モデルで同じテキストを使って推論
  torchrun --nproc_per_node=8 cosmos_predict1/diffusion/inference/text2world.py \
    --num_gpus 8 \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained-lora \
    --prompt "$UPSAMPLED_PROMPT" \
    --num_steps 35 \
    --video_save_folder outputs/comparison/prompt1/after \
    --video_save_name after \
    --disable_prompt_upsampler \
    --disable_guardrail

## 動画ファイルの送信
- rsync -avz outputs/cat_robot.mp4 10001249777@10.20.1.50:/Downloads

## データセット構築
- pip install yt-dlp
- nohup python3 download_raw_clips.py > output_download_raw_clips.log 2>&1 &

## nohup
- nohup python3 hoge.py > hoge.log 2>&1


# メモ
root@e9f825167add:/workspace# find datasets/posttrain_panda70m/cooking/videos/ -type f | wc -l
5258
root@e9f825167add:/workspace# find datasets/posttrain_panda70m/vehicle/videos/ -type f | wc -l
5433
root@e9f825167add:/workspace# find datasets/posttrain_panda70m/sports/videos/ -type f | wc -l
5715



## バッチ推論スクリプトinference/batch_inference.pyの使い方
- 機能
  - 単一条件に対し、複数シードの結果を出力
- テキスト条件付け
  - python3 inference/batch_inference.py \
  --model_type text2world \
  --prompt "a robot is surfing on the ocean" \
  --num_seeds 1 \
  --num_steps 10 \
  --fps 24 \
  --output_dir t2i_debug
- テキスト＋画像条件付け
  - python3 inference/batch_inference.py \
  --model_type video2world \
  --prompt "Denmaru, the tall round red character with arms extended, jumps straight up vertically while keeping arms fixed at sides. No rotation, tilt, or sideways movement. At the peak, briefly pauses, then lands smoothly on both feet and comes to a full stop. The camera is fixed, always showing the full body from the front. The motion is continuous and smooth without distortions." \
  --image_path "denmaru/denmaru3.jpg" \
  --num_seeds 10 \
  --num_steps 70 \
  --fps 24 \
  --guidance 9.0 \
  --output_dir denmaru3_1



# LoRA学習動くコード
export OUTPUT_ROOT=checkpoints
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_lora_example_cosmos_nemo_assets


# P08コード使い方
- デフォルト設定で実行 (r=8, iter=8000, bs=16, lr=1e-4, 720x1280)
  - python my_scripts/P08_run_continual_learning.py
- ハイパーパラメータを変更して実行する例
  - python my_scripts/P08_run_continual_learning.py \
    --lora_rank 16 \
    --max_iter 30 \
    --batch_size_per_gpu 1 \
    --learning_rate 1e-4 \
    --resolution 352 640
- ハイパラ
  - r = 8, 16, 32
  - lr = 5e-5, 1e-4, 3e-4
  - bs_per_gpu = 1, 2 (大きい方が学習速い)
  - max_iter = 5000 ~ 8000
  - resolution = [352,640] ([720,1280]のアスペクト比9:16は保つ, かつ16で割り切れないといけない)
