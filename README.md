# Cosmos

## 環境構築
- OS: Amazon Linux 2023
- EC2インスタンスはオンプレではなくAWSのクラウド上に立っているのでプロキシ設定は不要

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

## Devcontainerで、コンテナに入った後
- python依存パッケージをインストール
  - pip3 install --upgrade pip
  - pip3 install -r requirements.txt
- GPU環境の動作確認
  - python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"


## Dockerコマンドメモ
- docker ps -a
- docker stop (ID)
- docker rm (ID)


## 推論
- torchrun + DDP (offloadなし)
  - torchrun --nproc_per_node 8 \
  cosmos_predict1/diffusion/inference/text2world.py \
  --num_gpus 8 \
  --checkpoint_dir checkpoints \
  --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
  --prompt "$PROMPT" \
  --num_steps 30 \
  --video_save_folder outputs \
  --video_save_name splash_boy \
  --seed 0 \
  --fps 24 \
  --disable_guardrail
  - プロンプトアップサンプラーを切りたい場合
    - --disable_prompt_upsampler
