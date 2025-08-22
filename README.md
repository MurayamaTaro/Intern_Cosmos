# Intern Cosmos

## 環境構築
- VSCode拡張の「Dev containers」を入れる
- 画面左下の水色のボタンから、「コンテナで開く」を押す
  - cuda, pythonライブラリなど必要なものは全て入る
- ビルドに1時間ほどかかるので注意
- コード実行のためには以下フォルダが同じ階層に並んでいる必要あり
  - checkpoints/, cosmos_predict1/, dataset_panda70m/, dataset_ultravideo/, dataset_vript/, my_scripts/, .devcontainer/

## データセット

### VRipt
- https://github.com/mutonix/Vript
- 解像度720×1280、フレーム数121に整形済み
  - 解像度はアスペクト比を保ちつつセンタークロップ&スケーリングで処理
  - フレーム数は長尺動画を前から重複なしで121フレームずつ切り出し
- 付属カテゴリ名でドメイン分割済み
- テキスト埋め込みはしていない

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
- 以下コードは動作確認済み

### 準備（実行済みなので不要）
- モデル重みをダウンロード
  - PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World
  - 成果物はcheckpoints/に入る

### プロンプト埋め込み
-

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
  - num_stepsは拡散モデルのサンプリングステップのこと。大きくすると品質が上がるが計算時間も増える
  - guardrail, prompt_upsamplerは研究用途のため、またgpuを圧迫するため常に外す

### 追加学習
- python my_scripts/posttrain_single.py \
--dataset_path /workspace/dataset_panda70m/vehicle \
--dataset_name panda70m_vehicle \
--lora_rank 8 \
--max_iter 100 \
--batch_size_per_gpu 1 \
--learning_rate 1e-4 \
--scale 1.0 \
--grad_accum_iter 1 \
--seed 0
  - batch_size_per_gpuを増やすとOOMになる可能性あり, grad_accum_iterを増やすのが安全
  - scale（lora重みを何倍して足すか）は2, lora_rankは8, learning_rateは1e-4が標準的と思われる
  - max_iterは3~10エポックほど?(不明)

### 追加学習後LoRA重みでの推論
- python my_scripts/inference_orig_lora.py \
--inference_name test \
--experiment_name panda70m_vehicle_r8_iter100_bs1_accum1_scale1.0_lr1e-04_seed0 \
--prompt "A cinematic shot of a futuristic electric vehicle driving along a coastal road at sunset." \
--num_videos 1 \
--num_steps 50 \
--stages both
  - stagesにはoriginal（ベース重み）, lora（lora重み）, bothを指定できる

## 注意
- conda系の使用はNG(minicondaも)。ライセンスを取っていないため。
- Cosmosシステムは非常に厳密なのでcheckpointsの中のフォルダ名を変えるなどするとすぐエラーが出がちなため注意。
