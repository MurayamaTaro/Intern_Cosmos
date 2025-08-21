# Intern Cosmos


## データセット

### VRipt
- https://github.com/mutonix/Vript
- 解像度720×1280、フレーム数121に整形済み
  - 解像度はアスペクト比を保ちつつセンタークロップ&スケーリングで処理
  - フレーム数は長尺動画を前から重複なしで121フレームずつ切り出し
- 付属カテゴリ名でドメイン分割済み
- テキスト埋め込み済み

### UltraVideo
- https://github.com/xzc-zju/UltraVideo
- short_960をダウンロード
- 解像度720×1280、フレーム数121に整形済み
  - 解像度はアスペクト比を保ちつつセンタークロップ&スケーリングで処理
  - フレームが足りないときは補間で処理
- コーデック: H.264
- ドメイン分割はしていない
- テキスト埋め込みもしていない


## Cosmos (diffusion, text to video, 7B)
- https://github.com/nvidia-cosmos/cosmos-predict1
- https://docs.nvidia.com/cosmos/latest/

### 準備
- モデルパラメータをダウンロード
  - PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World

### 事前学習重みでの推論
- torchrun --nproc_per_node 8 \
cosmos_predict1/diffusion/inference/text2world.py \
--num_gpus 8 \
--checkpoint_dir checkpoints \
--diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
--prompt "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves." \
--num_steps 50 \
--video_save_folder outputs \
--video_save_name humanoid_robot \
--seed 0 \
--fps 24 \
--disable_guardrail \
--disable_prompt_upsampler

### 追加学習


## 注意
- conda使用はNG。ライセンスを取っていないため。
