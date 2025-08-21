#!/usr/bin/env bash
set -euo pipefail

# ===== 設定 =====
ROOT="/workspace"
LIST_DIR="${ROOT}/domain_lists"
OUT_DIR="${LIST_DIR}"                 # 同じ場所に *.headtail.csv を出す
PY="python3"
SPLITTER="${ROOT}/my_scripts/split_clip_timestamps.py"

DOMAINS=(vehicle sports animal food scenery tutorial gaming)

# bashの予約変数 SECONDS は使わない！
CLIP_SECONDS=5.2
MIN_FOR_TAIL=          # 空のまま＝完全2倍（ヘッド+テール）

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SPLITTER}" ]]; then
  echo "[ERROR] splitter not found: ${SPLITTER}" >&2
  exit 1
fi

for d in "${DOMAINS[@]}"; do
  in_csv="${LIST_DIR}/${d}.csv"
  out_csv="${OUT_DIR}/${d}.headtail.csv"

  if [[ ! -f "${in_csv}" ]]; then
    echo "[WARN] skip: ${in_csv} not found"
    continue
  fi

  echo "==> ${d}"
  in_lines=$( (wc -l < "${in_csv}") || echo 0 )
  echo "    input lines (with header): ${in_lines}"

  if [[ -n "${MIN_FOR_TAIL:-}" ]]; then
    "${PY}" "${SPLITTER}" \
      --in_csv  "${in_csv}" \
      --out_csv "${out_csv}" \
      --seconds "${CLIP_SECONDS}" \
      --min_for_tail "${MIN_FOR_TAIL}"
  else
    "${PY}" "${SPLITTER}" \
      --in_csv  "${in_csv}" \
      --out_csv "${out_csv}" \
      --seconds "${CLIP_SECONDS}"
  fi

  out_lines=$( (wc -l < "${out_csv}") || echo 0 )
  echo "    output lines (with header): ${out_lines}"

  in_n=$(( in_lines > 0 ? in_lines - 1 : 0 ))
  out_n=$(( out_lines > 0 ? out_lines - 1 : 0 ))
  expect=$(( in_n * 2 ))
  if [[ "${out_n}" -lt "${expect}" ]]; then
    echo "    [NOTE] out_n(${out_n}) < expect(${expect}). 5.2s未満の短尺が混じってる可能性あり。"
  else
    echo "    ok: doubled (${out_n} ~= ${expect})"
  fi
done

echo "ALL DONE."
