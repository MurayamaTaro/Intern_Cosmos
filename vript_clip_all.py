#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRipt 長編 → 121フレーム・720p クリップ（本番版）
- ドメイン: vript_meta/vript_long_videos_meta.json の "categories" をそのまま採用
  * ディレクトリ名はスラッグ化（howto_style 等）
- 切り出し: 前から重なり無しの 121 フレ窓、最大 15 クリップ/動画（端数切り捨て）
- 出力名: {domainSlug}_{videoID}_{windowIdx:05d}.{mp4/txt}
  * .txt は caption.content の1行のみ
- 変換: H.264(libx264) / CRF / preset 指定可、音声なし、1280x720（スケール→センタークロップ）
- 並列: 外側並列（ThreadPoolExecutor）。レジューム対応（既存OKはスキップ）
- ログ: グローバル（ドメイン外）に出力：
    dataset_vript/_logs/vript_process_log.csv
    dataset_vript/_logs/vript_failed.csv
  各ドメインには index.csv（クリップメタ）だけ残す
環境変数:
  VRIPT_WORKERS     … 外側並列数（デフォルト 32）
  FFMPEG_THREADS    … 各 ffmpeg の内部スレッド（デフォルト 1）
  ULTRA_PRESET      … x264 preset（デフォルト "faster"）
  ULTRA_CRF         … x264 CRF（デフォルト "23"）
  VRIPT_MAX_CLIPS   … 1動画あたりの最大クリップ数（デフォルト "15"）
  VRIPT_IDS         … カンマ区切りの video_id 指定（省略で全動画）
"""

import os, sys, json, math, time, csv, re, shlex, subprocess, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
from concurrent.futures import wait, FIRST_COMPLETED


# ルートと入力
ROOT = Path("dataset_vript")
IN_VIDEOS = ROOT / "vript_long_mp4"
IN_META   = ROOT / "vript_meta" / "vript_long_videos_meta.json"
IN_CAPS_DIR   = ROOT / "vript_captions" / "vript_long_videos_captions"
IN_CAPS_JSONL = ROOT / "vript_captions" / "vript_long_videos_captions.jsonl"

# 出力
CLIPS_ROOT = ROOT / "clips"
LOG_DIR = ROOT / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_LOG = LOG_DIR / "vript_process_log.csv"
GLOBAL_FAIL= LOG_DIR / "vript_failed.csv"

# パラメータ
WORKERS = int(os.getenv("VRIPT_WORKERS", "32"))
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", "1"))
X264_PRESET = os.getenv("ULTRA_PRESET", "faster")
X264_CRF    = os.getenv("ULTRA_CRF", "23")
MAX_CLIPS_PER_VIDEO = int(os.getenv("VRIPT_MAX_CLIPS", "15"))

# 安全センタークロップ
VF_CENTER = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

# ロック
log_lock = threading.Lock()
idx_locks = defaultdict(threading.Lock)  # domainごとの index.csv 用

def slugify(s: str) -> str:
    s = s.strip()
    # そのままのカテゴリもメタに残す。ディレクトリ名だけ安全化。
    sl = s.lower().replace("&","and")
    sl = re.sub(r"[^\w\s-]+","",sl, flags=re.UNICODE)
    sl = re.sub(r"[\s/]+","_",sl)
    sl = re.sub(r"_+","_",sl).strip("_")
    return sl or "other"

def out_dirs(domain_slug: str):
    base = CLIPS_ROOT / domain_slug
    vids = base / "videos"
    metas= base / "metas"
    vids.mkdir(parents=True, exist_ok=True)
    metas.mkdir(exist_ok=True)
    return base, vids, metas

def append_csv(path: Path, row: list, header: list, lock: threading.Lock = None):
    lock = lock or log_lock
    with lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)

def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "ignore").strip().splitlines()[-3:]
        e.cmd_str = " ".join(shlex.quote(c) for c in cmd) + (" :: " + " | ".join(tail) if tail else "")
        raise
    except subprocess.TimeoutExpired as e:
        e.cmd_str = " ".join(shlex.quote(c) for c in cmd) + " :: timeout"
        raise

def ffprobe_info(path: Path):
    # 速い2段階: (a) stream基本情報 (b) duration
    # どちらも "数秒" タイムアウト。デコードはしない（-count_frames禁止）
    width = height = fps = None
    nb_frames_hdr = None
    duration = None

    # a) stream基本
    try:
        p = run([
            "ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
            "-of","json", str(path)
        ], timeout=15)
        st = json.loads(p.stdout or b"{}").get("streams",[])
        if st:
            s = st[0]
            width  = s.get("width")
            height = s.get("height")
            # fps
            fr = s.get("avg_frame_rate") or s.get("r_frame_rate")
            if fr and fr!="0/0":
                try:
                    num,den = fr.split("/")
                    fps = float(num)/float(den) if float(den)!=0 else None
                except Exception:
                    fps = None
            # nb_frames（ヘッダにある場合のみ、信用できる範囲で使う）
            nf = s.get("nb_frames")
            if nf and str(nf).isdigit():
                nb_frames_hdr = int(nf)
    except Exception:
        pass

    # b) duration
    try:
        p = run([
            "ffprobe","-v","error","-show_entries","format=duration","-of","json", str(path)
        ], timeout=15)
        d = json.loads(p.stdout or b"{}").get("format",{}).get("duration")
        duration = float(d) if d is not None else None
        if duration is not None and (not math.isfinite(duration) or duration<=0):
            duration = None
    except Exception:
        pass

    return {
        "duration": duration,
        "frames": nb_frames_hdr,   # ヘッダにあれば数値、無ければ None
        "fps": fps,
        "width": width,
        "height": height,
    }

def build_common(src: Path):
    # mp4を明示（拡張子に依存しない）
    return [
        "ffmpeg","-hide_banner","-loglevel","error","-nostdin","-y",
        "-i", str(src), "-an",
        "-c:v","libx264","-preset",X264_PRESET,"-crf",X264_CRF,
        "-pix_fmt","yuv420p","-movflags","+faststart",
        "-threads",str(FFMPEG_THREADS),
        "-map_metadata","-1",
        "-f","mp4",
    ]

def verify_ok(path: Path):
    try:
        p = run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                 "-show_entries","stream=nb_read_frames,width,height","-of","json",str(path)], timeout=60)
        st=json.loads(p.stdout or b"{}").get("streams",[])
        if not st: return False
        s=st[0]; w=s.get("width"); h=s.get("height")
        n=s.get("nb_read_frames")
        fr=int(n) if n and str(n).isdigit() else None
        return (w==1280 and h==720 and fr==121)
    except Exception:
        return False

def load_meta_all():
    with IN_META.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_caption_list(video_id: str):
    # 1動画=1JSON（最優先）
    jpath = IN_CAPS_DIR / f"{video_id}_caption.json"
    if jpath.exists():
        try:
            j = json.loads(jpath.read_text(encoding="utf-8"))
            keys = sorted([k for k in j.get("data",{}).keys() if k.startswith(video_id)],
                          key=lambda x: x.split("-Scene-")[-1])
            caps = [j["data"][k]["caption"]["content"] for k in keys if "caption" in j["data"][k]]
            caps = [c.strip() for c in caps if c and c.strip()]
            return caps
        except Exception:
            pass
    # フォールバック：jsonlを走査
    caps=[]
    if IN_CAPS_JSONL.exists():
        with IN_CAPS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r=json.loads(line)
                    if r.get("meta",{}).get("video_id")==video_id:
                        c=(r.get("caption",{}) or {}).get("content","")
                        if c: caps.append(c.strip())
                except Exception:
                    continue
    return [c for c in caps if c]

def get_domain(meta_all: dict, vid: str):
    rec = meta_all.get(vid, {})
    cats = rec.get("categories") or ["other"]
    if isinstance(cats, list):
        raw = cats[0] if cats else "other"
    else:
        raw = str(cats) if cats else "other"
    return raw, slugify(raw), rec

def ensure_index(domain_slug: str):
    base, _, _ = out_dirs(domain_slug)
    idx_path = base / "index.csv"
    if not idx_path.exists():
        with idx_path.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["base","video_id","domain_slug","domain_raw","start_frame","fps","frames","width","height"])
    return idx_path

def cut_window(video_id: str, domain_slug: str, domain_raw: str,
               win_idx: int, start_frame: int, fps: float, caption: str):
    base, vids_dir, metas_dir = out_dirs(domain_slug)
    base_name = f"{domain_slug}_{video_id}_{win_idx:05d}"
    out_mp4 = vids_dir / f"{base_name}.mp4"
    out_txt = metas_dir / f"{base_name}.txt"
    tmp_mp4 = vids_dir / f"._tmp_{base_name}.mp4"

    # 既にOKならスキップ
    if out_mp4.exists() and out_txt.exists() and verify_ok(out_mp4):
        return ("skip", base_name, None)

    src = IN_VIDEOS / f"{video_id}.mp4"
    if not src.exists():
        return ("fail", base_name, "src_not_found")

    # caption（1行）
    if not caption:
        caption = video_id

    # 切り出し（精密シーク: 入力後 -ss）
    start_time = start_frame / fps if fps and fps>0 else 0.0
    if tmp_mp4.exists():
        tmp_mp4.unlink(missing_ok=True)
    cmd = build_common(src) + ["-ss", f"{start_time:.6f}", "-vf", VF_CENTER, "-frames:v","121", str(tmp_mp4)]
    run(cmd, timeout=10*60)

    if not verify_ok(tmp_mp4):
        tmp_mp4.unlink(missing_ok=True)
        return ("fail", base_name, "postcheck_failed")

    # メタ（caption.content のみ）
    metas_dir.mkdir(exist_ok=True)
    out_txt.write_text(caption+"\n", encoding="utf-8")

    # 原子的置換
    os.replace(tmp_mp4, out_mp4)
    return ("done", base_name, None)

def main():
    # 入力の収集
    ids_env = os.getenv("VRIPT_IDS","").strip()
    if ids_env:
        video_ids = [v.strip() for v in ids_env.split(",") if v.strip()]
    else:
        video_ids = sorted([p.stem for p in IN_VIDEOS.glob("*.mp4")])
    if not video_ids:
        print("[ERROR] no input videos found under", IN_VIDEOS, file=sys.stderr)
        sys.exit(1)

    meta_all = load_meta_all()

    # グローバルログを先に用意（空でもヘッダ作る）
    log_header  = ["ts","video_id","base","status","err","domain_slug","domain_raw","widx","start_frame","fps"]
    fail_header = ["video_id","domain_slug","widx","start_frame","err"]

    if not GLOBAL_LOG.exists():
        with GLOBAL_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(log_header)

    if not GLOBAL_FAIL.exists():
        with GLOBAL_FAIL.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fail_header)


    t0=time.time()
    print(f"[INFO] workers={WORKERS}, ffmpeg_threads={FFMPEG_THREADS}, preset={X264_PRESET}, crf={X264_CRF}")
    print(f"[INFO] videos: {len(video_ids)} total")
    print(f"[INFO] logs: {GLOBAL_LOG} , failed: {GLOBAL_FAIL}")

    # まず全動画の計画を作りつつ、ウィンドウの Future を段階的に投入（過大投入を避ける）
    INFLIGHT_LIMIT = int(os.getenv("INFLIGHT_LIMIT", str(WORKERS * 3)))
    inflight_limit = INFLIGHT_LIMIT
    futures = {}
    done_windows = 0
    total_windows = 0
    failed_videos = 0

    def schedule_windows(ex: ThreadPoolExecutor):
        nonlocal total_windows, failed_videos
        for vid in video_ids:
            domain_raw, domain_slug, rec = get_domain(meta_all, vid)
            src = IN_VIDEOS / f"{vid}.mp4"
            if not src.exists():
                # 動画レベルの失敗
                ts=datetime.utcnow().isoformat()
                append_csv(GLOBAL_LOG, [ts, vid, "", "fail_video", "not_found", domain_slug, domain_raw, "", "", ""], log_lock, )
                append_csv(GLOBAL_FAIL,[vid, domain_slug, "", "", "not_found"], fail_header, log_lock)
                failed_videos += 1
                continue

            try:
                info = ffprobe_info(src)
            except Exception as e:
                ts = datetime.utcnow().isoformat()
                err = "ffprobe_exception"
                append_csv(GLOBAL_LOG,  [ts, vid, "", "fail_video", err, domain_slug, domain_raw, "", "", ""], log_header, log_lock)
                append_csv(GLOBAL_FAIL, [vid, domain_slug, "", "", err],              fail_header, log_lock)
                continue

            # フレーム数の安全見積り
            F_hdr = info.get("frames")
            fps = info.get("fps") or 30.0
            dur = info.get("duration")
            if F_hdr and isinstance(F_hdr, int) and F_hdr > 0:
                F_est = F_hdr
            elif (dur and fps):
                F_est = int(dur * fps * 0.98)  # 2%クッション
            else:
                # 情報が取れない動画は動画ごとスキップ
                ts = datetime.utcnow().isoformat()
                append_csv(GLOBAL_LOG,  [ts, vid, "", "skip_video", "insufficient_probe", domain_slug, domain_raw, "", "", fps], log_header, log_lock)
                continue

            if F_est < 121:
                ts = datetime.utcnow().isoformat()
                append_csv(GLOBAL_LOG,  [ts, vid, "", "skip_video", "too_short", domain_slug, domain_raw, "", "", fps], log_header, log_lock)
                continue

            W = min(MAX_CLIPS_PER_VIDEO, F_est // 121)
            if W <= 0:
                ts = datetime.utcnow().isoformat()
                append_csv(GLOBAL_LOG,  [ts, vid, "", "skip_video", "no_window", domain_slug, domain_raw, "", "", fps], log_header, log_lock)
                continue

            # index.csv を準備
            idx_path = ensure_index(domain_slug)

            # キャプションをロード（シーン均等割り）
            caps = load_caption_list(vid)
            S = len(caps)

            for widx in range(W):
                start_frame = 121 * widx
                # 均等割当て
                caption = ""
                if S>0:
                    scene_idx = min(S-1, int(math.floor((widx / W) * S)))
                    caption = caps[scene_idx].strip()
                # 送信
                fut = ex.submit(cut_window, vid, domain_slug, domain_raw, widx, start_frame, fps, caption)
                futures[fut] = (vid, domain_slug, domain_raw, widx, start_frame, fps)
                total_windows += 1

                # 混み具合を制御
                while len(futures) >= inflight_limit:
                    yield

        # 送信完了
        return

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        scheduler = schedule_windows(ex)

        # 送信し終わったかどうかのフラグ
        all_submitted = False

        while True:
            # まず送信側を1ステップ進める（混み具合の制御は schedule_windows 内の inflight_limit と yield で維持）
            if not all_submitted:
                try:
                    next(scheduler)
                except StopIteration:
                    all_submitted = True

            if futures:
                # 1つでも終わったら取り出す（TimeoutErrorが出ない）
                done_set, _ = wait(list(futures.keys()), timeout=1.0, return_when=FIRST_COMPLETED)
                if not done_set:
                    continue  # まだ終わってないので次ループ
                for fut in done_set:
                    vid, domain_slug, domain_raw, widx, start_frame, fps = futures.pop(fut)
                    ts = datetime.utcnow().isoformat()
                    try:
                        status, base, err = fut.result()
                        append_csv(GLOBAL_LOG, [ts, vid, base or "", status, err or "", domain_slug, domain_raw, widx, start_frame, fps], log_header, log_lock)
                        if status == "fail":
                            append_csv(GLOBAL_FAIL, [vid, domain_slug, widx, start_frame, err or ""], fail_header, log_lock)
                        else:
                            # 成功/スキップ → index.csv を追記
                            idx_path = CLIPS_ROOT / domain_slug / "index.csv"
                            with idx_locks[domain_slug]:
                                with idx_path.open("a", newline="", encoding="utf-8") as f:
                                    w=csv.writer(f)
                                    w.writerow([base, vid, domain_slug, domain_raw, start_frame, fps, 121, 1280, 720])
                        done_windows += 1
                    except Exception as e:
                        msg = getattr(e, "cmd_str", str(e))
                        append_csv(GLOBAL_LOG, [ts, vid, "", "fail", msg, domain_slug, domain_raw, widx, start_frame, fps], log_header, log_lock)
                        append_csv(GLOBAL_FAIL, [vid, domain_slug, widx, start_frame, msg], fail_header, log_lock)
                        done_windows += 1
            else:
                # 何もin-flightがなく、かつ送信済みなら終了
                if all_submitted:
                    break
                time.sleep(0.1)

            # 進捗表示
            elapsed = time.time()-t0
            if done_windows>0 and done_windows % max(1,(WORKERS//2 or 1)) == 0:
                avg = elapsed / done_windows
                eta = avg * max(0, total_windows - done_windows)
                print(f"[{done_windows}/{total_windows}] avg={avg:.2f}s/clip, ETA~{eta/3600:.2f}h", flush=True)

    print("\n=== SUMMARY ===")
    print(f"videos={len(video_ids)}, windows_total={total_windows}, windows_done={done_windows}, elapsed={time.time()-t0:.1f}s")
    print(f"logs: {GLOBAL_LOG} , failed: {GLOBAL_FAIL}")
    print("Re-run with the same settings to resume (already-ok outputs are skipped).")

if __name__=="__main__":
    main()
