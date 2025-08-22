#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRipt: caption(.txt) → T5埋め込み（テスト版：指定ドメインから最大N件）
- 入力: dataset_vript/clips/{domain}/metas/*.txt（1行=caption.content）
- 出力: dataset_vript/clips/{domain}/t5_xxl/{same-basename}.pickle
  * 中身は token-wise hidden (seq_len x hidden_dim) を float16 で保存
"""

import os
import re
import csv
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

# ルート
ROOT = Path("dataset_vript")
CLIPS = ROOT / "clips"
LOGS  = ROOT / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)

def slug(s: str) -> str:
    s = s.strip().lower().replace("&","and")
    s = re.sub(r"[^\w\s\-]+","", s)
    s = re.sub(r"[\s/]+","_", s)
    return re.sub(r"_+","_", s).strip("_")

@torch.inference_mode()
def init_t5(model_name: str, max_length: int, cache_dir: str) -> Tuple[T5TokenizerFast, T5EncoderModel]:
    print(f"[INFO] loading T5: {model_name} (max_length={max_length})")
    tok = T5TokenizerFast.from_pretrained(model_name, model_max_length=max_length, cache_dir=cache_dir)
    enc = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        enc.to("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        print("[INFO] device = CUDA (bf16)")
    else:
        print("[WARN] CUDA not available → CPU実行")
    enc.eval()
    return tok, enc

@torch.inference_mode()
def encode_text(tok: T5TokenizerFast, enc: T5EncoderModel, text: str, max_length: int) -> np.ndarray:
    batch = tok.batch_encode_plus(
        [text],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
    )
    input_ids  = batch.input_ids.to(enc.device)
    attn_mask  = batch.attention_mask.to(enc.device)
    outputs    = enc(input_ids=input_ids, attention_mask=attn_mask)
    hidden     = outputs.last_hidden_state  # [1, L, H]
    length     = int(attn_mask.sum(dim=1).item())
    return hidden[0, :length].detach().cpu().to(torch.float32).numpy().astype(np.float16)

def pick_meta_files(domain: str, limit: int) -> List[Path]:
    metas_dir = CLIPS / domain / "metas"
    if not metas_dir.exists():
        raise FileNotFoundError(f"metas dir not found: {metas_dir}")
    files = sorted(metas_dir.glob("*.txt"))
    return files[:max(0, limit)]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="", help="例: autos_and_vehicles。空なら先頭のドメインを自動選択")
    ap.add_argument("--limit", type=int, default=5, help="処理件数（テスト用）")
    ap.add_argument("--model", type=str, default=os.getenv("T5_MODEL","google-t5/t5-11b"),
                    help="例: google-t5/t5-11b, google/t5-v1_1-xxl など")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cache-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    # ドメイン決定
    if args.domain:
        domain = args.domain.strip()
    else:
        domains = sorted([p.name for p in CLIPS.iterdir() if p.is_dir()])
        if not domains:
            raise RuntimeError(f"no domain dirs under {CLIPS}")
        domain = domains[0]
    print(f"[INFO] domain = {domain}")

    # 入力ファイル
    meta_files = pick_meta_files(domain, args.limit)
    if not meta_files:
        print("[WARN] no metas found. nothing to do.")
        return
    print(f"[INFO] files to process = {len(meta_files)}")

    # 出力ディレクトリ（固定で t5_xxl/ に保存）
    out_dir = CLIPS / domain / "t5_xxl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 索引ログ（テスト用）
    index_csv = LOGS / "t5_embed_test_index.csv"
    if not index_csv.exists():
        with index_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["domain","model","meta_path","pickle_path","tokens"])

    # モデル初期化
    tok, enc = init_t5(args.model, args.max_length, args.cache_dir)

    # ループ
    for meta_path in tqdm(meta_files, desc=f"T5 encode ({domain})"):
        base = meta_path.stem  # 例: autos_and_vehicles_-0sM8jwHi40_00000
        out_pkl = out_dir / f"{base}.pickle"
        if out_pkl.exists():
            # 既存はスキップ
            with index_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([domain, slug(args.model), str(meta_path), str(out_pkl), -1])
            continue

        # テキストは1行（caption.content）
        text = meta_path.read_text(encoding="utf-8").strip()
        if not text:
            with index_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([domain, slug(args.model), str(meta_path), "", 0])
            continue

        emb = encode_text(tok, enc, text, args.max_length)

        # 保存（pickle）
        with open(out_pkl, "wb") as f:
            pickle.dump(emb, f)

        # ログ
        with index_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([domain, slug(args.model), str(meta_path), str(out_pkl), emb.shape[0]])

    print("\n[OK] test finished.")
    print(f"[INFO] outputs under: {out_dir}")
    print(f"[INFO] index log: {index_csv}")

if __name__ == "__main__":
    main()
