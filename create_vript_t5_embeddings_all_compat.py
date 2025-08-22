# create_vript_t5_embeddings_all_compat.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRipt captions → T5 embeddings（panda70m互換）
- 出力: dataset_vript/ROOT/{domain}/t5_xxl/{same-basename}.pickle
- フォーマット: numpy.ndarray shape=(1, T, 1024), dtype=float32
- レジューム: 既存(1,*,1024)はスキップ
- OOM時: バッチ半減リトライ
"""

import os, re, csv, pickle
from pathlib import Path
from typing import List, Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast, T5EncoderModel

ROOT = Path("dataset_vript")
LOGS  = ROOT / "_logs"; LOGS.mkdir(parents=True, exist_ok=True)
INDEX_CSV = LOGS / "t5_embed_index.csv"
FAILED_CSV= LOGS / "t5_embed_failed.csv"

def slug(s: str) -> str:
    s = s.strip().lower().replace("&","and")
    s = re.sub(r"[^\w\s\-]+","", s)
    s = re.sub(r"[\s/]+","_", s)
    return re.sub(r"_+","_", s).strip("_")

def list_domains() -> List[str]:
    return sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

def list_meta_files(domain: str) -> List[Path]:
    return sorted((ROOT/domain/"metas").glob("*.txt"))

def ensure_logs():
    if not INDEX_CSV.exists():
        with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["domain","model","meta_path","pickle_path","tokens","status"])
    if not FAILED_CSV.exists():
        with FAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["domain","meta_path","reason"])

def is_ok_pickle(p: Path) -> bool:
    if not p.exists(): return False
    try:
        arr = pickle.load(open(p, "rb"))
        a = np.asarray(arr)
        return (a.ndim == 3) and (a.shape[0] == 1) and (a.shape[2] == 1024)
    except Exception:
        return False

def chunks(seq: List[Path], n: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

@torch.inference_mode()
def init_t5(model_name: str, max_length: int, cache_dir: str):
    print(f"[INFO] loading T5: {model_name} (max_length={max_length})")
    tok = T5TokenizerFast.from_pretrained(model_name, model_max_length=max_length, cache_dir=cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    enc = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)
    if torch.cuda.is_available():
        enc.to("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
        print("[INFO] device = CUDA (bf16)")
    else:
        print("[WARN] CUDA not available → CPU実行（遅い）")
    enc.eval()
    return tok, enc

@torch.inference_mode()
def encode_batch(tok, enc, texts: List[str], max_length: int) -> List[np.ndarray]:
    # バッチで走らせて各サンプル (1,T,1024) の np.float32 を返す
    batch = tok(
        texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt"
    )
    input_ids = batch["input_ids"].to(enc.device)
    attn      = batch["attention_mask"].to(enc.device)
    hs = enc(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [B, L, 1024]
    lens = attn.sum(dim=1).tolist()
    out = []
    for i, L in enumerate(lens):
        x = hs[i:i+1, :int(L), :]  # [1, T, 1024]
        out.append(x.detach().cpu().to(torch.float32).numpy())
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="", help="特定ドメインのみ処理（例: gaming）。空なら全ドメイン。")
    ap.add_argument("--model", type=str, default=os.getenv("T5_MODEL","google-t5/t5-11b"))
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("T5_BATCH","2")))
    ap.add_argument("--cache-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    ensure_logs()
    domains = [args.domain] if args.domain else list_domains()
    if not domains:
        raise RuntimeError(f"no domain dirs under {ROOT}")
    print(f"[INFO] domains = {domains}")

    tok, enc = init_t5(args.model, args.max_length, args.cache_dir)
    model_slug = slug(args.model)

    for domain in domains:
        metas = list_meta_files(domain)
        if not metas:
            print(f"[WARN] no metas in {domain}")
            continue
        out_dir = ROOT/domain/"t5_xxl"; out_dir.mkdir(parents=True, exist_ok=True)

        # レジューム：未処理だけ
        todo = []
        for mp in metas:
            out_p = out_dir / f"{mp.stem}.pickle"
            if not is_ok_pickle(out_p):
                todo.append((mp, out_p))
        print(f"[INFO] {domain}: metas={len(metas)}, to_process={len(todo)}")

        bs = max(1, args.batch_size)
        pbar = tqdm(total=len(todo), desc=f"T5 encode ({domain})", unit="file")

        i = 0
        while i < len(todo):
            # 現在のバッチ
            end = min(i + bs, len(todo))
            mpaths = [mp for (mp, _) in todo[i:end]]
            outs   = [op for (_,  op) in todo[i:end]]

            texts = []
            empty = []
            for mp in mpaths:
                try:
                    t = mp.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    t = ""
                texts.append(t); empty.append(t=="")

            # 空はログだけ
            for j, is_empty in enumerate(empty):
                if is_empty:
                    with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([domain, model_slug, str(mpaths[j]), "", 0, "empty"])
                    pbar.update(1)

            # 非空のサブバッチ実行
            sub_idx = [k for k, e in enumerate(empty) if not e]
            if sub_idx:
                try:
                    sub_texts = [texts[k] for k in sub_idx]
                    embeds = encode_batch(tok, enc, sub_texts, args.max_length)  # list of (1,T,1024)
                    for si, gi in enumerate(sub_idx):
                        out_p = outs[gi]
                        with open(out_p, "wb") as f:
                            pickle.dump(embeds[si], f)
                        with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([domain, model_slug, str(mpaths[gi]), str(out_p), embeds[si].shape[1], "ok"])
                        pbar.update(1)

                    i = end

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg:
                        bs = max(1, bs // 2); torch.cuda.empty_cache()
                        print(f"[WARN] CUDA OOM -> reduce batch-size to {bs} and retry.")
                        if bs == 1:
                            # 単品で順次処理
                            for j in sub_idx:
                                try:
                                    emb = encode_batch(tok, enc, [texts[j]], args.max_length)[0]
                                    with open(outs[j], "wb") as f:
                                        pickle.dump(emb, f)
                                    with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                        csv.writer(f).writerow([domain, model_slug, str(mpaths[j]), str(outs[j]), emb.shape[1], "ok"])
                                except Exception as e2:
                                    with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
                                        csv.writer(f).writerow([domain, str(mpaths[j]), f"error:{type(e2).__name__}"])
                                    with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                        csv.writer(f).writerow([domain, model_slug, str(mpaths[j]), "", 0, "fail"])
                                pbar.update(1)
                            i = end
                        # bs を下げたので同じ範囲を再試行（i は進めない）
                    else:
                        print(f"[WARN] runtime error: {e} -> fallback single loop")
                        for j in sub_idx:
                            try:
                                emb = encode_batch(tok, enc, [texts[j]], args.max_length)[0]
                                with open(outs[j], "wb") as f:
                                    pickle.dump(emb, f)
                                with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                    csv.writer(f).writerow([domain, model_slug, str(mpaths[j]), str(outs[j]), emb.shape[1], "ok"])
                            except Exception as e2:
                                with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
                                    csv.writer(f).writerow([domain, str(mpaths[j]), f"error:{type(e2).__name__}"])
                                with INDEX_CSV.open("a", newline="", encoding="utf-8") as f:
                                    csv.writer(f).writerow([domain, model_slug, str(mpaths[j]), "", 0, "fail"])
                            pbar.update(1)
                        i = end
            else:
                i = end

        pbar.close()
        print(f"[DONE] {domain}")

    print("\n[OK] all done.")
    print(f"[INFO] index:  {INDEX_CSV}")
    print(f"[INFO] failed: {FAILED_CSV}")

if __name__ == "__main__":
    main()
