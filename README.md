# Intern Cosmos

## 環境構築
- cd /data2/intern01/
- /data2/intern01/for_Cosmos/直下にdataset_{3種類}/とcheckpoints/を準備済み
  - それぞれ、データセットとモデル重みの実体が入っている
  - dataset_panda70m:91GB, dataset_vript:222GB, dataset_vript:40GB, checkpoints:303GB
- /data2/intern01/ で git clone https://github.com/MurayamaTaro/Intern_Cosmos.git
- /data2/intern01/Intern_Cosmos/のようなフォルダができる
- このフォルダにdataset_{3種類}/（重い）とcheckpoints/を移動(mvコマンド)
  - 同ファイルシステム内なのですぐに終わるはず
  - シンボリックリンクを貼っても良いかもしれないがコードの動作は未検証のため注意
- このフォルダをVSCodeで開く
- VSCode拡張機能の「Dev containers」を入れる
- 画面左下の水色のボタンを押し、「コンテナで開く」を押す
  - dockerコンテナのビルドが開始され、cuda, pythonライブラリなどが入る
  - **初回ビルドでは30分~1時間ほどかかるので注意**
  - 二回目からは一瞬で終わる
- 以下フォルダが同じ階層に並んでいることを確認（コード実行のために必要）
  - checkpoints/, cosmos_predict1/, my_scripts/


## データセット詳細

### VRipt
- https://github.com/mutonix/Vript
- 解像度720×1280、フレーム数121に整形済み
  - 解像度はアスペクト比を保ちつつセンタークロップ&スケーリングで処理
  - フレーム数は長尺動画を前から重複なしで121フレームずつ切り出し
- fps: 30 （元データ通り）
  - Cosmosの学習ではフレーム数が121であることが重要なので問題なし
  - fps24で推論すると少し遅く見える可能性あり（どちらでもok）
- 付属カテゴリ名でドメイン分割済み
- テキスト埋め込み済み

### UltraVideo
- https://github.com/xzc-zju/UltraVideo
- short_960をダウンロード
- 解像度720×1280、フレーム数121に整形済み
  - 解像度はアスペクト比を保ちつつセンタークロップ&スケーリングで処理
  - フレームが足りないときは補間で処理
- ドメイン分割はしていない
- テキスト埋め込みもしていない


## Cosmos (diffusion, text to video, 7B)
- https://github.com/nvidia-cosmos/cosmos-predict1
- https://docs.nvidia.com/cosmos/latest/
- 以下コードは動作確認済み

### 準備（実行済みなので作業不要）
- モデル重みをダウンロード
  - PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World
  - 成果物はcheckpoints/に入る

### プロンプト埋め込み
- 省略

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
  - huggingfaceのログインを求められたら、https://huggingface.co/でアカウント作成 -> 画面右上の自分のアイコンからAccess Tokensを押す -> Create new tokenを押す -> token typeをReadにしてToken nameを適当につけて（何でも良い）、トークンを取得 -> コマンドで huggingface-cli login, トークン入力
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
  - ハイパラについて
    - 総イテレーション数 max_iter = データ数 × エポック数 / 実効グローバルバッチ
    - batch_size_per_gpuを増やすとOutOfMemoryになる可能性あり -> grad_accum_iter（勾配累積）を増やすのが安全
      - 勾配累積：勾配を複数回溜めてからパラメータ更新するテクニックで、大バッチサイズを近似する方法としてよく使われる（大規模モデルではgpuメモリの制約によりバッチサイズを大きくできないため）
    - 実効バッチサイズ = #GPU(8) * batch_size_per_gpu * grad_accum_iter
      - 実効バッチサイズは16~64ぐらいが安定
    - エポック数：3~5エポックほど?
    - learning_rate：5e-5~2e-4 が標準的
    - lora_rank：8~32 が標準的
    - scale（lora重みを何倍して足すか）：1.5~2.5 が標準的
      - 大きくするならscaleかlrを下げて安定化必要
  - メモ
    - 計算時間は~
    - ログがたくさん出るがこれはCosmosの仕様
    - 損失値が負になることもあるが、Cosmosの損失関数はもともと負になりうるものを使っているため問題なし
    - 拡散モデルでは分類モデルのように損失値が急激に減少しない。学習時、毎回各データの別のタイムステップを見るため。
    - 分散学習はDDP（全gpuにモデル全体を置く）を採用している。gpuメモリを消費するが、FSDP（各gpuにモデルを分割して置く）ではクラッシュを回避できなかったため。

### 追加学習後LoRA重みでの推論
- python my_scripts/inference_orig_lora.py \
--inference_name test \
--experiment_name panda70m_vehicle_r8_iter100_bs1_accum1_scale1.0_lr1e-04_seed0 \
--prompt "A cinematic shot of a futuristic electric vehicle driving along a coastal road at sunset." \
--num_videos 1 \
--num_steps 50 \
--stages both
  - stagesにはoriginal（ベース重み）, lora（lora重み）, bothを指定できる

## 備考
- conda系の使用はNG(minicondaも)なため注意。ライセンスを取っていないため。
- Cosmosシステムは非常に厳密なのでcheckpointsの中のフォルダ名を変えるなどするとすぐエラーが出がちなため注意。
- UltraVideoを使う場合、プロンプト埋め込みにはvript加工時に使ったコードold/create_vript_t5_embeddings_all_compat.pyが参考になるかもしれない。
  - ドメイン分割については付属のメタ情報を使ってスクリプトを自作する必要あり
