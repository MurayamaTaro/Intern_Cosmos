# Cosmos

## 環境構築
- OS: Amazon Linux 2023
- EC2インスタンスはオンプレではなくAWSのクラウド上に立っているのでプロキシ設定は不要

## git設定
- git --version
- sudo dnf install -y git
- フォーク作成＆クロー
  - GitHub/GitLab上でフォーク作成
  - mkdir projects
  - cd projects
  - git clone https://github.com/nvidia-cosmos/cosmos-predict1.git
- フォーク確認
  - cd cosmos-predict1
  - git remote -v
- upstream (公式) リモートを追加
  - git remote add upstream https://github.com/nvidia-cosmos/cosmos-predict1.git
  - git remote -v
- 公式リポジトリの最新情報を取り込む
  - git fetch upstream
  - git branch -rで確認
- 実験用ブランチを切る（新ブランチを作成し切替）
  - git checkout -b experiment-1 upstream/main
  - git branchで確認
- コードを改造してコミット
  - touch tmp.py
  - git add .
  - git commit -m "add tmp.py"
- 実験ブランチをorigin（自分用写し）/実験ブランチにpush
  - git push -u origin experiment-1
- 実験ブランチをorigin/mainにpush
  - git checkout main
  - git merge experiment-1

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
