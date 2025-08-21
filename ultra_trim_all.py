#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UltraVideo clips_short_960 → 1280x720, 121 frames, no audio (production)
- 先に 1280x720 へ安全センタークロップ（黒幕なし）
  scale=1280:720:force_original_aspect_ratio=increase, crop=1280:720
- 121フレ以上の入力: 先頭121フレで切り出し（最速）
- 121フレ未満: boomerang → minterpolate(mci) → minterpolate(blend) → freeze の順でフォールバック
- H.264 (libx264), CRF 23, preset=slow, yuv420p, +faststart
- 並列処理、レジューム（既に正しく出力済みは自動スキップ）
- 失敗は failed.csv に記録。成功/スキップ/失敗は process_log.csv に追記（再実行で増分追記）

環境変数:
  ULTRA_WORKERS     … 外側の並列ワーカー数（推奨: 8〜32）
  FFMPEG_THREADS    … 各 ffmpeg の内部スレッド数（推奨: 1〜2）
  ULTRA_CRF         … デフォルト 23
  ULTRA_PRESET      … デフォルト "slow"
"""

import os, sys, time, json, math, shlex, subprocess, csv, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

IN_DIR  = Path("dataset_ultravideo/clips_short_960")
OUT_DIR = Path("dataset_ultravideo/clips_short_960_trimmed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 並列・エンコード設定
WORKERS = int(os.getenv("ULTRA_WORKERS", "16"))
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", "2"))
CRF = os.getenv("ULTRA_CRF", "23")
PRESET = os.getenv("ULTRA_PRESET", "slow")

# 進捗・ログ
LOG_CSV     = OUT_DIR / "process_log.csv"
FAILED_CSV  = OUT_DIR / "failed.csv"
VF_CENTER   = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

# ログ用ロック
log_lock = threading.Lock()

def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode("utf-8", "ignore").strip().splitlines()[-3:]
        e.cmd_str = " ".join(shlex.quote(c) for c in cmd) + (" :: " + " | ".join(msg) if msg else "")
        raise
    except subprocess.TimeoutExpired as e:
        e.cmd_str = " ".join(shlex.quote(c) for c in cmd) + " :: timeout"
        raise

def ffprobe_info(path: Path):
    """duration / frames / width / height / fps を取得（可能な範囲で）"""
    fmt_cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(path)]
    stm_cmd = ["ffprobe","-v","error","-count_frames","-select_streams","v:0",
               "-show_entries","stream=nb_read_frames,width,height,avg_frame_rate,r_frame_rate",
               "-of","json",str(path)]
    duration=None; frames=None; width=None; height=None; fps=None
    try:
        p=run(fmt_cmd,timeout=30); data=json.loads(p.stdout or b"{}")
        d=data.get("format",{}).get("duration")
        if d not in (None,"N/A"):
            duration=float(d);
            if not math.isfinite(duration) or duration<=0: duration=None
    except Exception: pass
    try:
        p=run(stm_cmd,timeout=60); data=json.loads(p.stdout or b"{}")
        sts=data.get("streams",[])
        if sts:
            s=sts[0]; width=s.get("width"); height=s.get("height")
            nrf=s.get("nb_read_frames"); frames=int(nrf) if nrf and str(nrf).isdigit() else None
            fr=s.get("avg_frame_rate") or s.get("r_frame_rate")
            if fr and fr!="0/0":
                try:
                    num,den=fr.split("/"); fps=float(num)/float(den) if float(den)!=0 else None
                except Exception: pass
    except Exception: pass
    if frames is None and (fps and duration): frames=int(round(fps*duration))
    return {"duration":duration,"frames":frames,"width":width,"height":height,"fps":fps}

def verify_ok(path: Path):
    """出力が 1280x720・121フレか？"""
    info = ffprobe_info(path)
    return (info["frames"]==121 and info["width"]==1280 and info["height"]==720)

def build_common(src: Path):
    """mp4出力を明示（-f mp4）、メタ削除、安定志向のx264設定"""
    return [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i",str(src),"-an",
        "-c:v","libx264","-preset",PRESET,"-crf",CRF,
        "-pix_fmt","yuv420p","-movflags","+faststart",
        "-threads",str(FFMPEG_THREADS),
        "-map_metadata","-1",
        "-f","mp4",  # コンテナ明示
    ]

# --- 各戦略 ---
def strat_cut121(src: Path, tmp: Path):
    """121フレ以上の入力 → 先頭121フレのみ（最速）"""
    vf = f"{VF_CENTER}"
    cmd = build_common(src) + ["-vf",vf,"-frames:v","121",str(tmp)]
    run(cmd, timeout=10*60)

def strat_boomerang(src: Path, tmp: Path):
    """正順＋逆順連結 → 121で切る（堅い & 速い）"""
    fc = f"[0:v]{VF_CENTER},split=2[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[outv]"
    cmd = build_common(src) + ["-filter_complex",fc,"-map","[outv]","-frames:v","121",str(tmp)]
    run(cmd, timeout=10*60)

def strat_mci(src: Path, tmp: Path, target_fps: float):
    """minterpolate (mci) → 121フレ"""
    vf = f"{VF_CENTER},minterpolate=fps={target_fps:.8f}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"
    cmd = build_common(src) + ["-vf",vf,"-frames:v","121",str(tmp)]
    run(cmd, timeout=10*60)

def strat_blend(src: Path, tmp: Path, target_fps: float):
    """minterpolate (blend) → 121フレ"""
    vf = f"{VF_CENTER},minterpolate=fps={target_fps:.8f}:mi_mode=blend"
    cmd = build_common(src) + ["-vf",vf,"-frames:v","121",str(tmp)]
    run(cmd, timeout=10*60)

def strat_freeze(src: Path, tmp: Path, need: int):
    """末尾フリーズで埋める（最後の砦）"""
    vf = f"{VF_CENTER}" if need<=0 else f"{VF_CENTER},tpad=stop={need}:stop_mode=clone"
    cmd = build_common(src) + ["-vf",vf,"-frames:v","121",str(tmp)]
    run(cmd, timeout=10*60)

# --- 1ファイル処理 ---
def process_one(basename: str):
    t0 = time.time()
    src = IN_DIR / basename
    if not src.exists():
        return {"name":basename,"ok":False,"status":"missing","strategy":None,"err":"not_found","sec":0.0}

    dst = OUT_DIR / basename
    tmp = OUT_DIR / f"._tmp_{Path(basename).stem}.mp4"  # .mp4拡張子を維持

    # 既にOK出力がある？ → スキップ（レジューム）
    if dst.exists() and verify_ok(dst):
        return {"name":basename,"ok":True,"status":"skip", "strategy":"exists","err":None,"sec":0.0}

    meta = ffprobe_info(src)
    orig_frames = meta["frames"] or 0
    duration = meta["duration"] or 0.0
    target_fps = 121.0/duration if duration>0 else 30.0
    need = max(0, 121 - orig_frames)

    # 戦略順：121 以上なら最速カット → 121未満なら boomerang → mci → blend → freeze
    strategies = []
    if orig_frames >= 121:
        strategies = [("cut121", lambda: strat_cut121(src,tmp))]
    else:
        strategies = [
            ("boomerang", lambda: strat_boomerang(src,tmp)),
            # ("mci",       lambda: strat_mci(src,tmp,target_fps)),
            # ("blend",     lambda: strat_blend(src,tmp,target_fps)),
            ("freeze",    lambda: strat_freeze(src,tmp,need)),
        ]

    last_err=None; used=None
    for name, fn in strategies:
        used = name
        try:
            if tmp.exists(): tmp.unlink(missing_ok=True)
            fn()
            if verify_ok(tmp):
                os.replace(tmp, dst)
                dt = time.time()-t0
                return {"name":basename,"ok":True,"status":"done","strategy":name,"err":None,"sec":dt}
            else:
                last_err = "postcheck_failed"
                tmp.unlink(missing_ok=True)
        except Exception as e:
            last_err = getattr(e,"cmd_str",str(e))
            tmp.unlink(missing_ok=True)
            # 次の戦略にフォールバック
    dt = time.time()-t0
    return {"name":basename,"ok":False,"status":"fail","strategy":used,"err":last_err,"sec":dt}

# --- ログユーティリティ ---
def append_csv(path: Path, row: list, header: list):
    exists = path.exists()
    with log_lock:
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)

def main():
    print(f"[INFO] workers={WORKERS}, ffmpeg_threads={FFMPEG_THREADS}, crf={CRF}, preset={PRESET}")
    print(f"[INFO] in ={IN_DIR.resolve()}")
    print(f"[INFO] out={OUT_DIR.resolve()}")
    # 入力一覧
    inputs = sorted([p.name for p in IN_DIR.glob("*.mp4")])
    total = len(inputs)
    if total==0:
        print("[ERROR] no .mp4 files under input dir")
        sys.exit(1)
    print(f"[INFO] total input clips: {total}")

    # 既存OKを数えて目安のETAを出す（都度更新もする）
    already_ok = sum(1 for n in inputs if (OUT_DIR/n).exists() and verify_ok(OUT_DIR/n))
    pending = total - already_ok
    print(f"[INFO] already ok: {already_ok}, pending: {pending}")

    hdr = ["ts","name","status","ok","strategy","sec","err"]
    t_start = time.time()
    done = 0; okcnt=0; failcnt=0

    # 並列実行
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_one, n): n for n in inputs}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            okcnt += int(r["ok"])
            failcnt += int(not r["ok"] and r["status"]=="fail")
            # ログ追記
            append_csv(LOG_CSV, [datetime.utcnow().isoformat(), r["name"], r["status"], r["ok"], r["strategy"], f"{r['sec']:.3f}", r["err"] or ""], hdr)
            if not r["ok"] and r["status"]=="fail":
                append_csv(FAILED_CSV, [r["name"], r["strategy"], r["err"] or ""], ["name","strategy","err"])

            # 進捗とETA
            elapsed = time.time()-t_start
            avg_sec = elapsed / done if done>0 else 0.0
            eta_sec = avg_sec * (total - done)
            print(f"[{done}/{total}] {r['name']} :: {r['status']} :: strat={r['strategy']} :: ok={r['ok']} :: {r['sec']:.1f}s "
                  f"|| avg={avg_sec:.1f}s/clip, ETA~{eta_sec/60:.1f} min", flush=True)

    print("\n=== SUMMARY ===")
    print(f"total={total}, ok={okcnt}, fail={failcnt}, elapsed={time.time()-t_start:.1f}s")
    print(f"logs: {LOG_CSV}, failed: {FAILED_CSV} (re-run will resume & retry fallbacks)")

if __name__=="__main__":
    main()
