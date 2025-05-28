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
- 単一推論
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

## 学習
- LoRA学習
  - デフォルトではmax_iter=1000
  - export OUTPUT_ROOT=checkpoints
  - torchrun --nproc_per_node=8 \
    -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_lora_example_cosmos_nemo_assets
- Full学習の場合
  - -- experiment=text2world_7b_example_cosmos_nemo_assets
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



## tmp
- PROMPT="In this captivating video, we are immersed in a sleek, futuristic laboratory setting, where a striking teal robot, reminiscent of a sophisticated industrial arm, takes center stage. The robot, equipped with a precision gripper, is poised on a polished metallic platform, its smooth surfaces reflecting the cool, artificial light that bathes the scene. The camera, positioned at a static angle, captures the robot's fluid movements as it gracefully extends its arm, reaching for a vibrant green cube resting on a nearby platform. With a delicate touch, the gripper envelops the cube, lifting it effortlessly into the air. The robot then navigates with precision, gliding towards a red platform, where it gently places the cube, showcasing its advanced dexterity. The background, adorned with a grid of metallic panels, enhances the industrial aesthetic, while the absence of human presence amplifies the focus on the robot's mechanical elegance and efficiency."
- torchrun --nproc_per_node=8 cosmos_predict1/diffusion/inference/text2world.py \
  --num_gpus 8 \
  --checkpoint_dir checkpoints \
  --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained \
  --prompt "$PROMPT" \
  --negative_prompt "" \
  --num_steps 60 \
  --video_save_folder outputs/debag \
  --video_save_name debag \
  --disable_prompt_upsampler \
  --disable_guardrail



PROMPT="In this captivating video, we are immersed in a sleek, futuristic laboratory setting, where a striking teal robot, reminiscent of a sophisticated industrial arm, takes center stage. The robot, equipped with a precision gripper, is poised on a polished metallic platform, its smooth surfaces reflecting the cool, artificial light that bathes the scene. The camera, positioned at a static angle, captures the robot's fluid movements as it gracefully extends its arm, reaching for a vibrant green cube resting on a nearby platform. With a delicate touch, the gripper envelops the cube, lifting it effortlessly into the air. The robot then navigates with precision, gliding towards a red platform, where it gently places the cube, showcasing its advanced dexterity. The background, adorned with a grid of metallic panels, enhances the industrial aesthetic, while the absence of human presence amplifies the focus on the robot's mechanical elegance and efficiency."
torchrun --nproc_per_node=8 cosmos_predict1/diffusion/inference/text2world.py \
  --num_gpus 8 \
  --checkpoint_dir            checkpoints       \
  --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained-lora \
  --prompt "${PROMPT}" \
  --num_steps 30\
  --video_save_folder outputs/demo --video_save_name demo \
  --disable_prompt_upsampler --disable_guardrail

