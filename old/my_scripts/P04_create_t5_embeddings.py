import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

# --- 設定 ---
# 【本番用】全カテゴリを対象
CATEGORIES = ['vehicle', 'sports', 'cooking']
# 【モデル設定】
PRETRAINED_MODEL_NAME = "google-t5/t5-11b"
CACHE_DIR = "checkpoints"
MAX_LENGTH = 512
# 【パフォーマンス設定】
# 複数のH100 GPUを活かすため、バッチサイズを大きく設定
# メモリ使用量に応じて調整してください
BATCH_SIZE = 128

# ベースパス
PROCESSED_DATA_DIR = Path('datasets/posttrain_panda70m')


@torch.inference_mode()
def init_t5(
    pretrained_model_name_or_path: str, max_length: int, cache_dir: str
) -> Tuple[T5TokenizerFast, T5EncoderModel]:
    """T5のトークナイザとエンコーダを初期化し、マルチGPUに対応させる"""
    print(f"Initializing T5 model: {pretrained_model_name_or_path}...")

    tokenizer = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, model_max_length=max_length, cache_dir=cache_dir
    )
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16
    )

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        if num_gpus > 1:
            print(f"Using torch.nn.DataParallel for {num_gpus} GPUs.")
            text_encoder = torch.nn.DataParallel(text_encoder)

        text_encoder.to("cuda")
        print("T5 model moved to CUDA.")
    else:
        print("Warning: CUDA not available. Running on CPU.")

    text_encoder.eval()
    return tokenizer, text_encoder


@torch.inference_mode()
def encode_text_prompt_batch(
    tokenizer: T5TokenizerFast, encoder: T5EncoderModel, prompts: List[str], max_length: int
) -> List[np.ndarray]:
    """テキストプロンプトのバッチをT5エンベディングに変換する"""
    device = next(encoder.parameters()).device

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
    )

    input_ids = batch_encoding.input_ids.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu().tolist()
    encoded_text_cpu = encoded_text.cpu().to(torch.float32)

    embeddings = []
    for i in range(len(prompts)):
        length = lengths[i]
        trimmed_embedding = encoded_text_cpu[i:i+1, :length].numpy().astype(np.float16)
        embeddings.append(trimmed_embedding)

    return embeddings


def create_t5_embeddings():
    """データセットの全キャプションからT5エンベディングを生成する"""

    tokenizer, text_encoder = init_t5(PRETRAINED_MODEL_NAME, MAX_LENGTH, CACHE_DIR)

    for category in CATEGORIES:
        print(f"\n--- Processing category: {category} ---")

        meta_dir = PROCESSED_DATA_DIR / category / 'metas'
        t5_xxl_dir = PROCESSED_DATA_DIR / category / 't5_xxl'
        t5_xxl_dir.mkdir(parents=True, exist_ok=True)

        # 処理対象のタスク（プロンプトと保存パス）をリストアップ
        tasks = []
        meta_files = sorted(list(meta_dir.glob('*.txt')))
        for meta_path in meta_files:
            pickle_path = t5_xxl_dir / meta_path.with_suffix('.pickle').name
            if not pickle_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                if prompt: # 空のプロンプトは除外
                    tasks.append({'prompt': prompt, 'path': pickle_path})

        if not tasks:
            print(f"All embeddings for {category} already exist. Skipping.")
            continue

        print(f"Found {len(tasks)} prompts to encode for {category}.")

        # バッチ処理
        for i in tqdm(range(0, len(tasks), BATCH_SIZE), desc=f"Generating embeddings for {category}"):
            batch_tasks = tasks[i:i + BATCH_SIZE]
            prompts_batch = [task['prompt'] for task in batch_tasks]

            embeddings_batch = encode_text_prompt_batch(tokenizer, text_encoder, prompts_batch, MAX_LENGTH)

            for j, task in enumerate(batch_tasks):
                with open(task['path'], "wb") as f:
                    pickle.dump(embeddings_batch[j], f)

    print("\nAll embeddings have been successfully generated.")


if __name__ == '__main__':
    create_t5_embeddings()
