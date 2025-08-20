#!/usr/bin/env bash
set -euo pipefail

cd /workspace/

# ===== 設定 =====
DOMAINS=(vehicle sports animal food scenery tutorial gaming)

V2D_CFG="/workspace/Panda-70M/dataset_dataloading/video2dataset/video2dataset/configs/panda70m.yaml"
LIST_DIR="/workspace/domain_lists"
OUT_ROOT="/workspace/datasets"

# 使うのは head/tail と retry のみ（plain {domain}.csv は不使用）
CSV_SUFFIX=".headtail.csv"

# リトライラウンド回数（安全に小刻みに）
MAX_RETRY_ROUNDS=3

# 作業ディレクトリ/ログ
TMPDIR="/workspace/.tmp"
LOGDIR="/workspace/logs"
mkdir -p "${TMPDIR}" "${LOGDIR}"
ulimit -n 65535 || true
export TMPDIR

# ===== ユーティリティ =====

# ヘッダ維持でシャッフル
shuffle_csv () {
  local in="$1" out="$2"
  { head -n1 "$in"; tail -n +2 "$in" | shuf; } > "$out"
}

# JSON群から「失敗 or mp4欠落」だけ抽出して retry CSV を作成（heredocでPython内蔵）
make_retry_csv () {
  local videos_dir="$1" out_csv="$2"
  python3 - "$videos_dir" "$out_csv" <<'PY'
import sys, json, csv, ast
from pathlib import Path

videos_dir = Path(sys.argv[1])
out_csv    = Path(sys.argv[2])

def parse_first_clip(s):
    try:
        x = ast.literal_eval(s)
        # 形: [[s,e]] / [(s,e)] / [[(s,e), ...]]
        if isinstance(x, list) and x:
            first = x[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                start, end = first
            elif isinstance(first, list) and first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
                start, end = first[0]
            else:
                return None
        elif isinstance(x, tuple) and len(x) >= 2:
            start, end = x[0], x[1]
        else:
            return None
        return f"[[\'{start}\', \'{end}\']]"  # Panda互換の書式で出す
    except Exception:
        return None

def parse_caption(s):
    # "['text']" or "text"
    try:
        x = ast.literal_eval(s)
        if isinstance(x, list) and x:
            return str(x[0])
        if isinstance(x, str):
            return x
    except Exception:
        pass
    return s

rows = []
for jp in videos_dir.rglob("*.json"):
    try:
        j = json.loads(jp.read_text(encoding="utf-8"))
    except Exception:
        continue
    status = (j.get("status") or "").lower()
    url    = j.get("url")
    clips  = j.get("clips")
    cap    = j.get("caption") or ""

    mp4p = jp.with_suffix(".mp4")
    need_retry = (status != "success") or (not mp4p.exists() or mp4p.stat().st_size == 0)
    if not need_retry:
        continue
    if not url or not clips:
        continue

    ts = parse_first_clip(clips)
    if not ts:
        continue

    caption = parse_caption(cap)
    rows.append((url, ts, caption))

# 重複排除（url+timestamp）
seen=set(); uniq=[]
for url,ts,cap in rows:
    k=(url,ts)
    if k in seen: continue
    seen.add(k); uniq.append((url,ts,cap))

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["url","timestamp","caption"])
    w.writerows(uniq)

print(f"wrote {len(uniq)} -> {out_csv}")
PY
}

# video2dataset 実行（共通）
run_v2d () {
  local csv="$1" out_dir="$2" log="$3"
  video2dataset \
    --url_list="${csv}" \
    --output_folder="${out_dir}" \
    --url_col="url" \
    --caption_col="caption" \
    --clip_col="timestamp" \
    --save_additional_columns="[matching_score,desirable_filtering,shot_boundary_detection]" \
    --config="${V2D_CFG}" \
    --tmp_dir="${TMPDIR}" \
    --incremental_mode="incremental" \
    > "${log}" 2>&1
}

# ===== ドメイン実行 =====
run_domain () {
  local dom="$1"
  local in_csv
  local shuf_csv
  local out_dir="${OUT_ROOT}/${dom}/videos"
  [[ -d "$out_dir" ]] || mkdir -p "$out_dir"

  # --- 優先順位: retry.csv > headtail.csv ---
  if [[ -f "${LIST_DIR}/${dom}.retry.csv" ]]; then
    in_csv="${LIST_DIR}/${dom}.retry.csv"
  elif [[ -f "${LIST_DIR}/${dom}.headtail.csv" ]]; then
    in_csv="${LIST_DIR}/${dom}.headtail.csv"
  else
    echo "[WARN] no CSV for ${dom}" ; return
  fi

  shuf_csv="${in_csv}.shuf"
  shuffle_csv "$in_csv" "$shuf_csv"

  local log="${LOGDIR}/${dom}_$(date +%Y%m%d_%H%M%S).log"

  echo "[INFO] Running ${dom} using $(basename "$in_csv")"
  video2dataset \
    --url_list="${shuf_csv}" \
    --output_folder="${out_dir}" \
    --url_col="url" \
    --caption_col="caption" \
    --clip_col="timestamp" \
    --save_additional_columns="[matching_score,desirable_filtering,shot_boundary_detection]" \
    --config="${V2D_CFG}" \
    --tmp_dir="${TMPDIR}" \
    --incremental_mode="incremental" \
    > "${log}" 2>&1

  sleep 90
  local n_mp4
  n_mp4=$(find "${out_dir}" -type f -name '*.mp4' | wc -l | tr -d ' ')
  echo "    ${dom}: mp4 files = ${n_mp4}"
}


# ===== 実行 =====
for d in "${DOMAINS[@]}"; do
  run_domain "${d}"
done

echo "ALL DONE."
