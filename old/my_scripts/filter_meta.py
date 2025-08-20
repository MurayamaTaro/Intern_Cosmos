# my_scripts/filter_meta.py
import sys, csv
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

import re, ast, pandas as pd
from pathlib import Path

SRC = Path("panda70m_train_full.csv")  # or .csv.gz
OUT = Path("domain_lists"); OUT.mkdir(exist_ok=True)

def parse_dur_s(ts):
    # ts: "[['0:00:04.004','0:00:13.113'], ...]" ← シングル/ダブル混在に強い
    try:
        pairs = ast.literal_eval(ts)  # JSONじゃない行も読める
        s, e = pairs[0]
        def to_s(x):
            h, m, s = x.split(":")
            return int(h)*3600 + int(m)*60 + float(s)
        return max(0.0, to_s(e) - to_s(s))
    except Exception:
        return 0.0

def is_single_shot(x):
    try:
        sb = ast.literal_eval(x)
        return len(sb) == 1
    except Exception:
        # 形式不明なら除外してもいいが、ここでは一旦True（後段で長さ>=5秒で担保）
        return True

DOMAINS = [
    ("vehicle",  r"(vehicle|car|truck|bus|motorcycle|bike|bicycle|scooter|train|tram|subway|airplane|boat|ship|helicopter|rally|racing)"),
    ("sports",   r"(soccer|football|basketball|baseball|tennis|volleyball|hockey|golf|cricket|rugby|marathon|running|ski|snowboard|surf|boxing|mma|wrestling|badminton|table tennis|skate|cycling|gymnastics)"),
    ("animal",   r"(dog|cat|horse|bird|cow|elephant|lion|tiger|bear|fish|shark|whale|dolphin|zebra|giraffe|panda|monkey|puppy|kitten)"),
    ("food",     r"(cooking|recipe|baking|grilling|kitchen|chef|dish|pasta|soup|salad|cake|bread|bbq|sushi)"),
    ("scenery",  r"(landscape|mountain|beach|ocean|sea|river|forest|skyline|sunset|sunrise|timelapse|waterfall|nature|aerial|drone)"),
    ("tutorial", r"(tutorial|how to|step by step|guide|diy|lesson|learn|tips|tricks)"),
    ("gaming",   r"(gameplay|let's play|walkthrough|speedrun|minecraft|fortnite|roblox|apex|genshin|valorant|3d render|unreal engine|unity|blender|shader|ray tracing)"),
]
DOMAINS = [(n, re.compile(pat, re.I)) for n, pat in DOMAINS]

assigned = set()
writers = {n: open(OUT/f"{n}.csv", "w", encoding="utf-8") for n,_ in DOMAINS}
for w in writers.values():
    w.write("videoID,url,timestamp,caption,matching_score,desirable_filtering,shot_boundary_detection\n")

usecols = ["videoID","url","timestamp","caption","matching_score",
           "desirable_filtering","shot_boundary_detection"]

# 重要: engine="python", on_bad_lines="skip", low_memory=False
reader = pd.read_csv(
    SRC,
    usecols=usecols,
    chunksize=250_000,
    engine="python",
    on_bad_lines="skip",
    dtype=str,            # いったん全部文字列で受ける
    keep_default_na=False,
    # low_memory=False,
)

for chunk in reader:
    # 念のため列名トリム（スペース混入対策）
    chunk.columns = chunk.columns.str.strip()

    # 欠落チャンク（極稀に起きる）をスキップ
    if "caption" not in chunk.columns or "timestamp" not in chunk.columns:
        continue

    # 5秒以上
    dur = chunk["timestamp"].map(parse_dur_s)
    chunk = chunk[dur >= 5.0]

    # desirableのみ（列が文字列でもOK）
    if "desirable_filtering" in chunk.columns:
        chunk = chunk[chunk["desirable_filtering"].str.contains("desirable", case=False, na=False)]

    # 単一ショット優先（可能なら）
    if "shot_boundary_detection" in chunk.columns:
        chunk = chunk[chunk["shot_boundary_detection"].map(is_single_shot)]

    if chunk.empty:
        continue

    # ドメイン優先順で排他的に割当
    for name, rx in DOMAINS:
        mask = chunk["caption"].str.contains(rx, na=False)
        sub = chunk[mask & ~chunk["videoID"].isin(assigned)]
        if not sub.empty:
            sub.to_csv(writers[name], header=False, index=False)
            assigned.update(sub["videoID"].tolist())

for w in writers.values():
    w.close()
print("done")
