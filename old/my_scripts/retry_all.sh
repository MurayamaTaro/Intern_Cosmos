#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
LIST_DIR="${ROOT}/domain_lists"
OUT_ROOT="${ROOT}/datasets"
V2D_CFG="${ROOT}/Panda-70M/dataset_dataloading/video2dataset/video2dataset/configs/panda70m.yaml"
PY="python3"
RETRY_MAKER="${ROOT}/my_scripts/make_retry_csv_full.py"
DOMAINS=(vehicle sports animal food scenery tutorial gaming)

TMPDIR="${ROOT}/.tmp"
LOGDIR="${ROOT}/logs"
mkdir -p "${TMPDIR}" "${LOGDIR}"
export TMPDIR
ulimit -n 65535 || true

# stdout は「リトライCSVのフルパス」だけ返す。ログはファイルへ。
make_retry_csv () {
  local dom="$1"
  local videos_dir="${OUT_ROOT}/${dom}/videos"
  local src_csv="${LIST_DIR}/${dom}.headtail.csv"
  local retry_csv="${LIST_DIR}/${dom}.retry.csv"
  local mklog="${LOGDIR}/${dom}_make_retry_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "${src_csv}" || ! -d "${videos_dir}" ]]; then
    echo "" ; return 2
  fi

  # Pythonの出力はログに吸わせて、stdoutは汚さない
  if ! ${PY} "${RETRY_MAKER}" \
        --videos_dir "${videos_dir}" \
        --src_csv    "${src_csv}" \
        --out_csv    "${retry_csv}" \
        > "${mklog}" 2>&1; then
    echo "" ; return 1
  fi

  # ヘッダのみ＝対象なし
  if [[ "$(wc -l < "${retry_csv}")" -le 1 ]]; then
    echo "" ; return 3
  fi

  echo "${retry_csv}"
  return 0
}

run_retry_domain () {
  local dom="$1"
  local retry_csv
  retry_csv="$(make_retry_csv "${dom}")" || true
  if [[ -z "${retry_csv}" ]]; then
    echo "[INFO] ${dom}: retry対象なし（または生成失敗）"
    return 0
  fi

  local out_dir="${OUT_ROOT}/${dom}/videos"
  local log="${LOGDIR}/${dom}_retry_$(date +%Y%m%d_%H%M%S).log"
  local nrows=$(( $(wc -l < "${retry_csv}") - 1 ))
  echo "==> RETRY ${dom}: ${nrows} rows -> ${out_dir} (log: ${log})"

  video2dataset \
    --url_list="${retry_csv}" \
    --output_folder="${out_dir}" \
    --url_col="url" \
    --caption_col="caption" \
    --clip_col="timestamp" \
    --config="${V2D_CFG}" \
    --tmp_dir="${TMPDIR}" \
    --incremental_mode="incremental" \
    > "${log}" 2>&1

  # ドメイン間に小休止（超安全運転）
  sleep 90
}

for d in "${DOMAINS[@]}"; do
  run_retry_domain "${d}"
done

echo "ALL RETRIES DONE."
