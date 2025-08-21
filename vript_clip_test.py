#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRipt 長編→クリップ（テスト版：指定2本）
- ドメイン: vript_meta/vript_long_videos_meta.json の "categories" をそのまま利用
  * ディレクトリ名はスラッグ化（howto_style 等）。index.csv には元カテゴリも保持
- 切り出し: 121フレーム窓、重なりなし、最大 15 クリップ/動画（端数切り捨て）
- 変換: スケール→センタークロップで 1280x720、無音、H.264（libx264）
- 命名: {domainSlug}_{videoID}_{windowIdx:05d}.mp4（テキストも同名ベース）
- 出力: dataset_vript/clips/{domainSlug}/{videos,metas}/
- ログ: 進捗、process_log.csv、failed.csv（テスト版でも本番同様の形式）

環境変数:
  VRIPT_WORKERS   … 外側並列数（デフォルト 8）
  FFMPEG_THREADS  … 各ffmpeg内部スレッド（デフォルト 1）
  ULTRA_PRESET    … x264 preset（デフォルト "faster"）
  ULTRA_CRF       … x264 CRF（デフォルト "23"）
"""

import os, sys, json, math, time, csv, shlex, subprocess, argparse, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ルート
ROOT = Path("dataset_vript")
IN_VIDEOS = ROOT / "vript_long_mp4"
IN_META   = ROOT / "vript_meta" / "vript_long_videos_meta.json"
IN_CAPS_DIR = ROOT / "vript_captions" / "vript_long_videos_captions"
IN_CAPS_JSONL = ROOT / "vript_captions" / "vript_long_videos_captions.jsonl"

# パラメータ
WORKERS = int(os.getenv("VRIPT_WORKERS", "8"))
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", "1"))
X264_PRESET = os.getenv("ULTRA_PRESET", "faster")
X264_CRF    = os.getenv("ULTRA_CRF", "23")
MAX_CLIPS_PER_VIDEO = int(os.getenv("VRIPT_MAX_CLIPS", "15"))

# 出力ログ
def out_dirs(domain_slug: str):
    base = ROOT / "clips" / domain_slug
    vids = base / "videos"
    metas= base / "metas"
    base.mkdir(parents=True, exist_ok=True)
    vids.mkdir(exist_ok=True)
    metas.mkdir(exist_ok=True)
    return vids, metas, base

def log_paths(domain_slug: str):
    _, _, base = out_dirs(domain_slug)
    return base / "process_log.csv", base / "failed.csv", base / "index.csv"

VF_CENTER = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"

def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "ignore").strip().splitlines()[-3:]
        e.cmd_str = " ".join(shlex.quote(c) for c in cmd) + (" :: " + " | ".join(tail) if tail else "")
        raise

def ffprobe_info(path: Path):
    dur=None; frames=None; fps=None; w=None; h=None
    # duration
    p = run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(path)])
    d=json.loads(p.stdout or b"{}").get("format",{}).get("duration")
    try:
        dur=float(d);
        if not math.isfinite(dur) or dur<=0: dur=None
    except Exception: dur=None
    # stream
    p = run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
             "-show_entries","stream=nb_read_frames,width,height,avg_frame_rate,r_frame_rate",
             "-of","json",str(path)])
    st=json.loads(p.stdout or b"{}").get("streams",[])
    if st:
        s=st[0]
        w=s.get("width"); h=s.get("height")
        nrf=s.get("nb_read_frames")
        frames=int(nrf) if nrf and str(nrf).isdigit() else None
        fr=s.get("avg_frame_rate") or s.get("r_frame_rate")
        if fr and fr!="0/0":
            try:
                num,den=fr.split("/")
                fps=float(num)/float(den) if float(den)!=0 else None
            except Exception: fps=None
    # 近似
    if frames is None and fps and dur: frames=int(round(fps*dur))
    if fps is None and frames and dur and dur>0: fps=float(frames)/dur
    return {"duration":dur,"frames":frames,"fps":fps,"width":w,"height":h}

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("&","and")
    s = re.sub(r"[^\w\s-]+","",s, flags=re.UNICODE)
    s = re.sub(r"[\s/]+","_",s)
    s = re.sub(r"_+","_",s).strip("_")
    return s or "other"

def load_meta():
    with IN_META.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_caption_map_for_video(vid: str):
    # 1動画=1JSON（最優先）
    path = IN_CAPS_DIR / f"{vid}_caption.json"
    if path.exists():
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
            # data は { f"{vid}-Scene-001": {...}, ... }
            keys = sorted([k for k in j.get("data",{}).keys() if k.startswith(vid)], key=lambda x: x.split("-Scene-")[-1])
            return j.get("meta",{}).get("num_clips", len(keys)), [j["data"][k]["caption"]["content"] for k in keys]
        except Exception:
            pass
    # フォールバック: jsonl から拾う
    caps=[]
    if IN_CAPS_JSONL.exists():
        with IN_CAPS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r=json.loads(line)
                    if r.get("meta",{}).get("video_id")==vid:
                        caps.append(r.get("caption",{}).get("content","").strip())
                except Exception:
                    continue
    caps = [c for c in caps if c]
    return len(caps), caps

def verify_ok(out_path: Path):
    try:
        p = run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                 "-show_entries","stream=nb_read_frames,width,height","-of","json",str(out_path)], timeout=60)
        st=json.loads(p.stdout or b"{}").get("streams",[])
        if not st: return False
        s=st[0]; w=s.get("width"); h=s.get("height")
        n=s.get("nb_read_frames")
        fr=int(n) if n and str(n).isdigit() else None
        return (w==1280 and h==720 and fr==121)
    except Exception:
        return False

def build_common(src: Path):
    return [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i", str(src), "-an",
        "-c:v","libx264","-preset",X264_PRESET,"-crf",X264_CRF,
        "-pix_fmt","yuv420p","-movflags","+faststart",
        "-threads",str(FFMPEG_THREADS),
        "-map_metadata","-1",
        "-f","mp4",
    ]

def cut_one_window(src: Path, start_time: float, out_tmp: Path):
    # 入力後 -ss（精密シーク）→ 121 frames 固定 → 安全センタークロップ
    vf = VF_CENTER
    cmd = build_common(src) + ["-ss", f"{start_time:.6f}", "-vf", vf, "-frames:v","121", str(out_tmp)]
    run(cmd, timeout=10*60)

def process_one_clip(vid: str, meta_rec: dict, win_idx: int, start_frame: int, fps: float,
                     caps_list: list, total_wins: int, domain_label: str, domain_slug: str):
    # 命名
    base_name = f"{domain_slug}_{vid}_{win_idx:05d}"
    vids_dir, metas_dir, base_dir = out_dirs(domain_slug)
    out_mp4 = vids_dir / f"{base_name}.mp4"
    out_txt = metas_dir / f"{base_name}.txt"
    tmp_mp4 = vids_dir / f"._tmp_{base_name}.mp4"

    # 既存OKならスキップ
    if out_mp4.exists() and out_txt.exists() and verify_ok(out_mp4):
        return ("skip", base_name, None)

    # キャプション（均等割り当て）
    S = len(caps_list)
    if S>0 and total_wins>0:
        scene_idx = min(S-1, int(math.floor((win_idx / total_wins) * S)))
        caption = caps_list[scene_idx].strip()
    else:
        # フォールバック: タイトル
        caption = (meta_rec.get("title") or "").strip() or vid

    # 切り出し
    start_time = start_frame / fps if fps and fps>0 else 0.0
    if tmp_mp4.exists(): tmp_mp4.unlink()
    cut_one_window(IN_VIDEOS / f"{vid}.mp4", start_time, tmp_mp4)
    if not verify_ok(tmp_mp4):
        tmp_mp4.unlink(missing_ok=True)
        raise RuntimeError("postcheck_failed")

    # メタ書き出し（caption.contentのみ）
    metas_dir.mkdir(exist_ok=True)
    out_txt.write_text(caption+"\n", encoding="utf-8")

    # 原子的リネーム
    os.replace(tmp_mp4, out_mp4)
    return ("done", base_name, None)

def process_one_video(vid: str, meta_all: dict):
    src = IN_VIDEOS / f"{vid}.mp4"
    if not src.exists():
        return {"video":vid, "ok":False, "err":"not_found", "clips":0}

    meta_rec = meta_all.get(vid, {})
    cats = meta_rec.get("categories") or ["other"]
    if isinstance(cats, list): cat_raw = cats[0] if cats else "other"
    else: cat_raw = str(cats) if cats else "other"
    domain_slug = slugify(cat_raw)

    # キャプションロード
    S, caps_list = load_caption_map_for_video(vid)

    # ffprobe
    info = ffprobe_info(src)
    F = info["frames"] or 0
    fps = info["fps"] or 30.0
    if F < 121:
        return {"video":vid, "ok":True, "err":"too_short", "clips":0}

    W = min(MAX_CLIPS_PER_VIDEO, F // 121)
    if W <= 0:
        return {"video":vid, "ok":True, "err":"no_window", "clips":0}

    # ログファイル
    log_csv, failed_csv, index_csv = log_paths(domain_slug)
    header = ["ts","video_id","base","status","err","domain","domain_raw","widx","start_frame","fps"]
    fh = open(log_csv, "a", newline="", encoding="utf-8")
    fw = csv.writer(fh)
    if fh.tell()==0: fw.writerow(header)

    # index.csv（1行/クリップ のメタ）
    if not index_csv.exists():
        with index_csv.open("w", newline="", encoding="utf-8") as fidx:
            w=csv.writer(fidx); w.writerow(
                ["base","video_id","domain_slug","domain_raw","start_frame","fps","frames","width","height"]
            )

    done=0; fail=0
    # 並列に各ウィンドウを処理
    tasks=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for widx in range(W):
            start_frame = 121*widx
            tasks.append(ex.submit(
                process_one_clip, vid, meta_rec, widx, start_frame, fps, caps_list, W, cat_raw, domain_slug
            ))
        for fut in as_completed(tasks):
            widx = tasks.index(fut)  # 簡易でOK（テスト用）
            base=None
            try:
                status, base, err = fut.result()
                ts=datetime.utcnow().isoformat()
                fw.writerow([ts, vid, base, status, "", domain_slug, cat_raw, widx, 121*widx, fps])
                fh.flush()
                if status=="done":
                    # index.csv 追記
                    with index_csv.open("a", newline="", encoding="utf-8") as fidx:
                        w=csv.writer(fidx); w.writerow([base, vid, domain_slug, cat_raw, 121*widx, fps, 121, 1280, 720])
                    done += 1
                elif status=="skip":
                    done += 1
            except Exception as e:
                ts=datetime.utcnow().isoformat()
                msg=getattr(e,"cmd_str",str(e))
                fw.writerow([ts, vid, base or "", "fail", msg, domain_slug, cat_raw, widx, 121*widx, fps])
                fh.flush()
                with open(failed_csv, "a", newline="", encoding="utf-8") as ff:
                    w=csv.writer(ff)
                    if Path(failed_csv).stat().st_size==0:
                        w.writerow(["video_id","domain","widx","start_frame","err"])
                    w.writerow([vid, domain_slug, widx, 121*widx, msg])
                fail += 1
    fh.close()
    return {"video":vid, "ok": fail==0, "err": None if fail==0 else f"{fail} fails", "clips": done}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="", help="comma-separated video IDs (e.g. \"--2IG_rxfUM,--4M68p_Loc\")")
    args = ap.parse_args()

    # IDs 決定
    if args.ids.strip():
        vids = [v.strip() for v in args.ids.split(",") if v.strip()]
    else:
        # 引数無しなら mp4 から先頭2本
        vids = sorted([p.stem for p in IN_VIDEOS.glob("*.mp4")])[:2]

    if len(vids)==0:
        print("[ERROR] no input videos found.")
        sys.exit(1)

    meta_all = load_meta()

    t0=time.time()
    print(f"[INFO] workers={WORKERS}, ffmpeg_threads={FFMPEG_THREADS}, preset={X264_PRESET}, crf={X264_CRF}")
    print(f"[INFO] testing videos: {vids}")

    total_done=0; total_fail=0
    for vid in vids:
        print(f"\n[BEGIN] {vid}")
        res = process_one_video(vid, meta_all)
        print(f"[END]   {vid} :: ok={res['ok']} :: err={res['err']} :: clips={res['clips']}")
        if not res["ok"]: total_fail += 1
        total_done += res["clips"]

    dt=time.time()-t0
    print(f"\n=== TEST SUMMARY ===")
    print(f"videos={len(vids)}, clips_done={total_done}, fails={total_fail}, elapsed={dt:.1f}s "
          f"({dt/max(1,total_done):.2f}s/clip avg)")

if __name__=="__main__":
    main()
