#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRipt: 全ドメインの caption(.txt) を T5 エンコーダで埋め込み → pickle
- 入力: dataset_vript/clips/{domain}/metas/*.txt（1行=caption.content）
- 出力: dataset_vript/clips/{domain}/t5_xxl/{same-basename}.pickle
  * 中身: token-wise hidden (seq_len x hidden_dim) を float16 で保存
- 設計: 並列スレッドなし（GPUでバッチ処理）、OOM時はバッチを自動半減してリトライ
- レジューム: 既存pickleはスキップ
- ログ: dataset_vript/_logs/t5_embed_index.csv / t5_embed_failed.csv
"""

import os
import re
import csv
import pickle
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

# ルート
ROOT = Path("dataset_vript")
CLIPS = ROOT / "clips"
LOGS  = ROOT / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)

INDEX_CSV = LOGS / "t5_embed_index.csv"
FAILED_CSV= LOGS / "t5_embed_failed.csv"

def slug(s: str) -> str:
    s = s.strip().lower().replace("&","and")
    s = re.sub(r"[^\w\s\-]+","", s)
    s = re.sub(r"[\s/]+","_", s)
    return re.sub(r"_+","_", s).strip("_")

def list_domains() -> List[str]:
    if not CLIPS.exists():
        return []
    return sorted([p.name for p in CLIPS.iterdir() if p.is_dir()])

def list_meta_files(domain: str) -> List[Path]:
    metas_dir = CLIPS / domain / "metas"
    if not metas_dir.exists():
        return []
    return sorted(metas_dir.glob("*.txt"))

def ensure_logs():
    if not INDEX_CSV.exists():
        with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["domain","model","meta_path","pickle_path","tokens","status"])
    if not FAILED_CSV.exists():
        with FAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["domain","meta_path","reason"])

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
        print("[WARN] CUDA not available → CPU実行（遅い）")
    enc.eval()
    return tok, enc

def chunks(seq: List[Path], n: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

@torch.inference_mode()
def encode_batch_texts(tok: T5TokenizerFast, enc: T5EncoderModel, texts: List[str], max_length: int
                      ) -> List[np.ndarray]:
    """
    バッチを一度に推論し、各サンプルごとに [Li, H] を返す。
    """
    batch = tok.batch_encode_plus(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
    )
    input_ids = batch.input_ids.to(enc.device)
    attn_mask = batch.attention_mask.to(enc.device)

    outputs = enc(input_ids=input_ids, attention_mask=attn_mask)
    hidden  = outputs.last_hidden_state  # [B, L, H]
    lens    = attn_mask.sum(dim=1).tolist()
    out_list = []
    for i, L in enumerate(lens):
        arr = hidden[i, :int(L)].detach().cpu().to(torch.float32).numpy().astype(np.float16)
        out_list.append(arr)
    return out_list

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="", help="特定ドメインのみ処理（例: autos_and_vehicles）。空なら全ドメイン。")
    ap.add_argument("--model", type=str, default=os.getenv("T5_MODEL","google-t5/t5-11b"),
                    help="例: google-t5/t5-11b, google/t5-v1_1-xxl など")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("T5_BATCH","2")),
                    help="GPUメモリに合わせて調整（大きすぎるとOOM→自動で半減リトライ）")
    ap.add_argument("--cache-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    ensure_logs()

    # 対象ドメイン
    domains = [args.domain] if args.domain else list_domains()
    if not domains:
        raise RuntimeError(f"no domain dirs under {CLIPS}")
    print(f"[INFO] domains = {domains}")

    # モデル
    tok, enc = init_t5(args.model, args.max_length, args.cache_dir)
    model_slug = slug(args.model)

    # 各ドメインを順に処理（スレッド並列なし）
    for domain in domains:
        metas = list_meta_files(domain)
        if not metas:
            print(f"[WARN] no metas in domain: {domain}")
            continue

        out_dir = CLIPS / domain / "t5_xxl"
        out_dir.mkdir(parents=True, exist_ok=True)

        # レジューム：未処理だけ抽出
        todo = []
        for mp in metas:
            base = mp.stem
            out_pkl = out_dir / f"{base}.pickle"
            if not out_pkl.exists():
                todo.append((mp, out_pkl))
        print(f"[INFO] {domain}: total metas={len(metas)}, to_process={len(todo)}")

        bs = max(1, args.batch_size)

        pbar = tqdm(total=len(todo), desc=f"T5 encode ({domain})", unit="file")
        i = 0
        while i < len(todo):
            # 現在のバッチを構成
            end = min(i + bs, len(todo))
            batch_paths = [mp for (mp, _) in todo[i:end]]
            batch_outs  = [op for (_,  op) in todo[i:end]]
            texts = []
            empty_mask = []
            for mp in batch_paths:
                try:
                    t = mp.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception as e:
                    t = ""
                texts.append(t)
                empty_mask.append(1 if t=="" else 0)

            try:
                # 空のものはスキップ扱いにして先にログだけ書く
                for j, is_empty in enumerate(empty_mask):
                    if is_empty:
                        with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([domain, model_slug, str(batch_paths[j]), "", 0, "empty"])
                        pbar.update(1)
                # 非空だけを推論するためのサブバッチを作る
                sub_idx = [k for k, is_empty in enumerate(empty_mask) if not is_empty]
                if sub_idx:
                    sub_texts = [texts[k] for k in sub_idx]
                    # 推論
                    embeds = encode_batch_texts(tok, enc, sub_texts, args.max_length)

                    # 保存＆ログ
                    for idx_in_sub, gi in enumerate(sub_idx):
                        out_pkl = batch_outs[gi]
                        with open(out_pkl, "wb") as f:
                            pickle.dump(embeds[idx_in_sub], f)
                        with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([domain, model_slug, str(batch_paths[gi]), str(out_pkl), embeds[idx_in_sub].shape[0], "ok"])
                        pbar.update(1)

                # 次のバッチへ
                i = end

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    # OOM → バッチ半減 & キャッシュ解放
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    print(f"[WARN] CUDA OOM -> reduce batch-size to {bs} and retry.")
                    if bs == 1:
                        # 1でも落ちる場合は個別に失敗扱いしてスキップ
                        for j, is_empty in enumerate(empty_mask):
                            if not is_empty:
                                with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
                                    csv.writer(f).writerow([domain, str(batch_paths[j]), "cuda_oom"])
                                with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                    csv.writer(f).writerow([domain, model_slug, str(batch_paths[j]), "", 0, "fail_oom"])
                                pbar.update(1)
                        i = end  # このバッチを諦めて次へ
                    # bs を下げたので while ループの先頭から “同じ i〜end 範囲” を再試行
                else:
                    # その他の実行時エラー：各サンプルを個別に処理して原因を切り分け
                    print(f"[WARN] runtime error in batch, fallback to single-item loop. err={e}")
                    for j, is_empty in enumerate(empty_mask):
                        if is_empty:
                            continue
                        try:
                            emb = encode_batch_texts(tok, enc, [texts[j]], args.max_length)[0]
                            out_pkl = batch_outs[j]
                            with open(out_pkl, "wb") as f:
                                pickle.dump(emb, f)
                            with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([domain, model_slug, str(batch_paths[j]), str(out_pkl), emb.shape[0], "ok"])
                            pbar.update(1)
                        except Exception as e2:
                            with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([domain, str(batch_paths[j]), f"error:{type(e2).__name__}"])
                            with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([domain, model_slug, str(batch_paths[j]), "", 0, "fail"])
                            pbar.update(1)
                    i = end
            except Exception as e:
                # トークナイズ/IOなどの致命的でないエラーはサンプル失敗として継続
                print(f"[WARN] unexpected error, skip current batch. err={e}")
                for j, is_empty in enumerate(empty_mask):
                    if not is_empty:
                        with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([domain, str(batch_paths[j]), f"error:{type(e).__name__}"])
                        with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([domain, model_slug, str(batch_paths[j]), "", 0, "fail"])
                        pbar.update(1)
                i = end

        pbar.close()
        print(f"[DONE] {domain}")

    print("\n[OK] all done.")
    print(f"[INFO] index:  {INDEX_CSV}")
    print(f"[INFO] failed: {FAILED_CSV}")

if __name__ == "__main__":
    main()
