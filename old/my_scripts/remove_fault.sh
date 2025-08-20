DOMAIN=sports
OUT=/workspace/datasets/${DOMAIN}/videos

# 失敗 or mp4が無いサンプルだけを掃除
python - <<'PY'
import json, os, glob, sys
DOMAIN=sys.argv[1]; OUT=f"/workspace/datasets/{DOMAIN}/videos"
removed=0
for j in glob.glob(os.path.join(OUT, "*", "*.json")):
    try:
        with open(j) as f: m=json.load(f)
    except Exception:
        # 壊れjsonは対象
        m={"status":"failed_to_download"}
    key = m.get("key", os.path.splitext(os.path.basename(j))[0])
    mp4 = os.path.join(os.path.dirname(j), f"{key}.mp4")
    status = m.get("status","")
    need = (status!="success") or (not os.path.exists(mp4))
    if need:
        for ext in (".json",".txt",".mp4"):
            p=os.path.join(os.path.dirname(j), f"{key}{ext}")
            if os.path.exists(p):
                try: os.remove(p); removed+=1
                except: pass
print(f"[CLEAN] removed files: {removed}")
PY
# 中間シャードを一度クリア
rm -rf "${OUT}/_tmp"
