#!/usr/bin/env python3
import argparse, csv, json, ast
from pathlib import Path

def parse_hms_to_ms(s: str) -> int:
    # 'H:MM:SS.mmm' or 'HH:MM:SS.mmm' → ms(int)
    h, m, sec = s.split(":")
    ms = round((int(h)*3600 + int(m)*60 + float(sec)) * 1000.0)
    return int(ms)

def canon_ts_key(ts_field: str):
    """
    Panda系: "[['0:00:06.000','0:00:54.000']]" / "[('00:00:06.000','00:00:54.000')]"
    などを許容。先頭ペアだけ使って (start_ms, end_ms) を返す。
    フォーマットが変でも None を返してスキップ。
    """
    try:
        x = ast.literal_eval(ts_field)
        if isinstance(x, list):
            first = x[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                s, e = first
            elif isinstance(first, list) and first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
                s, e = first[0]
            else:
                return None
        elif isinstance(x, tuple) and len(x) >= 2:
            s, e = x[0], x[1]
        else:
            return None
        return (parse_hms_to_ms(str(s)), parse_hms_to_ms(str(e)))
    except Exception:
        return None

def collect_failed_keys(videos_dir: Path, check_mp4: bool=True):
    """
    datasets/{domain}/videos 以下の .json を走査。
    - status != 'success' を失敗扱い
    - または mp4 不在/サイズ0 も失敗扱い（check_mp4=True のとき）
    → (url, start_ms, end_ms) の集合を返す
    """
    failed = set()
    for jp in videos_dir.rglob("*.json"):
        try:
            j = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = (j.get("status") or "").lower()
        url = j.get("url")
        clips = j.get("clips")
        if not url or not clips:
            continue

        # 先頭クリップだけ見る（head/tail運用なので1ペア想定）
        key_ts = canon_ts_key(clips)
        if not key_ts:
            continue

        need_retry = status != "success"
        if check_mp4:
            mp4 = jp.with_suffix(".mp4")
            if (not mp4.exists()) or (mp4.stat().st_size == 0):
                need_retry = True

        if need_retry:
            failed.add((url, key_ts[0], key_ts[1]))
    return failed

def main():
    ap = argparse.ArgumentParser(description="Make retry CSV while preserving ALL columns from source CSV.")
    ap.add_argument("--videos_dir", required=True, help="datasets/{domain}/videos")
    ap.add_argument("--src_csv",    required=True, help="domain_lists/{domain}.headtail.csv（元の入力CSV）")
    ap.add_argument("--out_csv",    required=True, help="出力: domain_lists/{domain}.retry.csv（全列保持）")
    ap.add_argument("--no_check_mp4", action="store_true", help="mp4不在チェックを無効化（statusのみで判断）")
    args = ap.parse_args()

    vids = Path(args.videos_dir)
    src  = Path(args.src_csv)
    out  = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    failed = collect_failed_keys(vids, check_mp4=(not args.no_check_mp4))
    print(f"[INFO] failed keys: {len(failed)}")

    # src CSV をストリームしながら一致した行だけ書く（全列保持）
    kept = 0
    with src.open("r", newline="", encoding="utf-8") as fin, \
         out.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        if "url" not in fieldnames or "timestamp" not in fieldnames:
            raise SystemExit("[ERROR] src_csv には 'url' と 'timestamp' 列が必要です")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            url = row.get("url")
            ts_key = canon_ts_key(row.get("timestamp",""))
            if not url or not ts_key:
                continue
            if (url, ts_key[0], ts_key[1]) in failed:
                writer.writerow(row)
                kept += 1

    print(f"[INFO] wrote {kept} rows -> {out}")

if __name__ == "__main__":
    main()
