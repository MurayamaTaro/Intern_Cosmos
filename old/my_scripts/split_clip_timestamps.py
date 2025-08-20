#!/usr/bin/env python3
import csv, sys, argparse, ast
from pathlib import Path

def parse_hms(s: str) -> float:
    # 'H:MM:SS.mmm' を秒に
    h, m, s = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def fmt_hms(t: float) -> str:
    # 秒 -> 'H:MM:SS.mmm'（ミリ精度、ゼロ詰め）
    if t < 0: t = 0.0
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = t
    return f"{h}:{m:02d}:{s:06.3f}"

def load_first_pair(ts_field: str):
    """
    Panda系の timestamp フィールドは "[['HH:MM:SS.xxx','HH:MM:SS.xxx']]" のような
    Pythonリテラルっぽい文字列で来ることが多い。ast.literal_eval で安全に読む。
    常に最初のペアだけ使う（通常は1ペア/行前提）。
    """
    try:
        pairs = ast.literal_eval(ts_field)
        if isinstance(pairs, list) and len(pairs) >= 1 and len(pairs[0]) == 2:
            return pairs[0][0], pairs[0][1]
    except Exception:
        pass
    return None, None

def build_ts_str(start: float, end: float) -> str:
    return f"[[\'{fmt_hms(start)}\', \'{fmt_hms(end)}\']]"

def main():
    ap = argparse.ArgumentParser(description="Duplicate each row into head/tail 5.2s clips.")
    ap.add_argument("--in_csv", required=True, help="入力CSV（Panda互換）")
    ap.add_argument("--out_csv", required=True, help="出力CSV")
    ap.add_argument("--seconds", type=float, default=5.2, help="切り出し秒数（既定5.2）")
    ap.add_argument("--min_for_tail", type=float, default=None,
                    help="末尾も作るための最小長（秒）。未指定なら重なっても両方作る。例: 10.4 を指定すると dur>=10.4 の時だけ2行。")
    ap.add_argument("--chunk_rows", type=int, default=200000, help="チャンクサイズ（行）")
    ap.add_argument("--id_suffix", action="store_true", default=True,
                    help="videoIDに _head/_tail サフィックスを付ける（既定ON）")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        # 必須列の存在チェック
        required = ["videoID","url","timestamp","caption"]
        for r in required:
            if r not in fieldnames:
                print(f"[ERROR] 必須列 {r} が見つからない", file=sys.stderr); sys.exit(1)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        buf = []
        def flush():
            nonlocal buf
            if buf:
                writer.writerows(buf)
                buf = []

        for i, row in enumerate(reader, 1):
            s_str, e_str = load_first_pair(row["timestamp"])
            if not s_str or not e_str:
                # フォーマット異常はスキップ
                continue
            S = parse_hms(s_str); E = parse_hms(e_str)
            dur = max(0.0, E - S)
            if dur < args.seconds:
                # 5.2s 未満はスキップ（video2dataset側で失敗しがち）
                continue

            head_end = min(S + args.seconds, E)
            tail_start = max(E - args.seconds, S)

            make_tail = True
            if args.min_for_tail is not None and dur < args.min_for_tail:
                make_tail = False

            # 1) HEAD
            row_h = dict(row)
            row_h["timestamp"] = build_ts_str(S, head_end)
            if args.id_suffix and "videoID" in row_h:
                row_h["videoID"] = f"{row_h['videoID']}_head"
            buf.append(row_h)

            # 2) TAIL（条件付き）
            if make_tail:
                row_t = dict(row)
                row_t["timestamp"] = build_ts_str(tail_start, E)
                if args.id_suffix and "videoID" in row_t:
                    row_t["videoID"] = f"{row_t['videoID']}_tail"
                buf.append(row_t)

            if i % args.chunk_rows == 0:
                flush()

        flush()

if __name__ == "__main__":
    main()
