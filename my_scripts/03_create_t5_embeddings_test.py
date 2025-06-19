import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

# --- 設定 ---
# 【テスト用】処理対象のカテゴリとファイル数
CATEGORIES = ['vehicle']
PROCESS_LIMIT = 5

# 【モデル設定】
# 使用するT5モデル
PRETRAINED_MODEL_NAME = "google-t5/t5-11b"
# モデルのダウンロード先キャッシュディレクトリ
CACHE_DIR = "checkpoints"
# テキストの最大長
MAX_LENGTH = 512

# ベースパス
PROCESSED_DATA_DIR = Path('datasets/posttrain_panda70m')


@torch.inference_mode()
def init_t5(
    pretrained_model_name_or_path: str, max_length: int, cache_dir: str
) -> Tuple[T5TokenizerFast, T5EncoderModel]:
    """T5のトークナイザとエンコーダを初期化して返す"""
    print(f"Initializing T5 model: {pretrained_model_name_or_path}...")
    print("This may take a while for the first time to download the model...")

    tokenizer = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, model_max_length=max_length, cache_dir=cache_dir
    )
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16
    )

    if torch.cuda.is_available():
        text_encoder.to("cuda")
        print("T5 model moved to CUDA.")
    else:
        print("Warning: CUDA not available. Running on CPU.")

    text_encoder.eval()
    return tokenizer, text_encoder


@torch.inference_mode()
def encode_text_prompt(
    tokenizer: T5TokenizerFast, encoder: T5EncoderModel, prompt: str, max_length: int
) -> np.ndarray:
    """単一のテキストプロンプトをT5エンベディングに変換する"""
    batch_encoding = tokenizer.batch_encode_plus(
        [prompt],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
    )

    input_ids = batch_encoding.input_ids.to(encoder.device)
    attn_mask = batch_encoding.attention_mask.to(encoder.device)

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    # 不要なパディング部分をトリムして容量を節約
    length = attn_mask.sum(dim=1).cpu().item()
    trimmed_embedding = encoded_text[0, :length].cpu().to(torch.float32).numpy().astype(np.float16)

    return trimmed_embedding


def create_t5_embeddings_for_test():
    """【テスト用】データセットのキャプションからT5エンベディングを生成する"""

    # T5モデルを初期化（この処理は一度だけ行われる）
    tokenizer, text_encoder = init_t5(PRETRAINED_MODEL_NAME, MAX_LENGTH, CACHE_DIR)

    for category in CATEGORIES:
        print(f"\n--- Processing category: {category} (TEST MODE: {PROCESS_LIMIT} files) ---")

        # --- パスの設定 ---
        meta_dir = PROCESSED_DATA_DIR / category / 'metas'
        t5_xxl_dir = PROCESSED_DATA_DIR / category / 't5_xxl'
        t5_xxl_dir.mkdir(parents=True, exist_ok=True)

        # --- 処理対象のファイルリストを取得 ---
        meta_files = sorted(list(meta_dir.glob('*.txt')))
        files_to_process = meta_files[:PROCESS_LIMIT]

        print(f"Found {len(files_to_process)} files to process.")

        for meta_path in tqdm(files_to_process, desc=f"Generating embeddings for {category}"):
            pickle_path = t5_xxl_dir / meta_path.with_suffix('.pickle').name

            if pickle_path.exists():
                continue

            # キャプションを読み込む
            with open(meta_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()

            # エンベディングを計算
            embedding = encode_text_prompt(tokenizer, text_encoder, prompt, MAX_LENGTH)

            # pickleファイルとして保存
            with open(pickle_path, "wb") as f:
                pickle.dump(embedding, f)

    print("\nTest finished.")


if __name__ == '__main__':
    create_t5_embeddings_for_test()
