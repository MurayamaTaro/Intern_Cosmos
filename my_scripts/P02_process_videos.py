import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import imageio
import decord
import numpy as np

# --- 設定 ---
# 【本番用】全カテゴリを対象
CATEGORIES = ['vehicle', 'sports', 'cooking']
# PROCESS_LIMITは削除

# モデルが要求する仕様
TARGET_FPS = 24
TARGET_FRAMES = 121
TARGET_RESOLUTION = (1280, 720)

# ベースパス
BASE_DATA_DIR = Path('datasets/Panda-70M')
PROCESSED_DATA_DIR = Path('datasets/posttrain_panda70m')

def process_videos():
    """
    【本番用】mapping.csvに基づき、全動画を前処理する。
    decordを使用し、フレーム数不足の統計情報を記録・表示する。
    """
    decord.bridge.set_bridge('native')

    # --- 統計情報用の変数を初期化 ---
    total_processed = 0
    total_padded = 0
    total_skipped = 0
    category_stats = {}

    for category in CATEGORIES:
        print(f"--- Processing category: {category} ---")

        # --- カテゴリごとの統計情報を初期化 ---
        cat_processed = 0
        cat_padded = 0
        cat_skipped = 0

        # --- パスの設定 ---
        mapping_csv_path = PROCESSED_DATA_DIR / category / 'mapping.csv'
        original_clips_dir = BASE_DATA_DIR / 'clips' / category
        output_videos_dir = PROCESSED_DATA_DIR / category / 'videos'

        output_videos_dir.mkdir(parents=True, exist_ok=True)

        if not mapping_csv_path.exists():
            print(f"Warning: mapping.csv not found for {category}. Skipping.")
            continue

        mapping_df = pd.read_csv(mapping_csv_path)

        for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=f"Processing videos for {category}"):
            new_base_name = row['new_base_name']
            original_filename = row['original_video_filename']

            source_video_path = original_clips_dir / original_filename
            output_video_path = output_videos_dir / f"{new_base_name}.mp4"

            if output_video_path.exists():
                continue
            if not source_video_path.exists():
                cat_skipped += 1
                continue

            try:
                vr = decord.VideoReader(str(source_video_path), ctx=decord.cpu(0))
                processed_frames = []

                num_frames_to_read = min(len(vr), TARGET_FRAMES)

                for i in range(num_frames_to_read):
                    frame = vr[i]
                    numpy_frame = frame.asnumpy()
                    resized_frame = cv2.resize(numpy_frame, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
                    processed_frames.append(resized_frame)

                if not processed_frames:
                    cat_skipped += 1
                    continue

                num_existing_frames = len(processed_frames)
                if num_existing_frames < TARGET_FRAMES:
                    # 【統計】フレーム不足をカウント
                    cat_padded += 1
                    last_frame = processed_frames[-1]
                    padding_count = TARGET_FRAMES - num_existing_frames
                    processed_frames.extend([last_frame] * padding_count)

                writer = imageio.get_writer(
                    output_video_path, fps=TARGET_FPS, codec='libx264', quality=8
                )
                for frame in processed_frames:
                    writer.append_data(frame)
                writer.close()

                cat_processed += 1

            except (decord.DECORDError, Exception) as e:
                # エラーが発生したファイルはスキップとしてカウント
                cat_skipped += 1
                # nohupで実行するため、エラーログは標準エラーに出力され、ファイルに記録される
                # print(f"\nError processing {original_filename}: {e}") # ログが大量になるためコメントアウト
                continue

        # --- カテゴリごとのサマリーを表示 ---
        print(f"\n--- Summary for category: {category} ---")
        print(f"Successfully processed: {cat_processed} files")
        print(f"Padded (short) files: {cat_padded} files")
        print(f"Skipped (missing/error): {cat_skipped} files")
        print("-" * 30)

        # 全体統計に加算
        total_processed += cat_processed
        total_padded += cat_padded
        total_skipped += cat_skipped

    # --- 最終サマリーを表示 ---
    print("\n" + "="*40)
    print("           FINAL SUMMARY")
    print("="*40)
    print(f"Total files processed successfully: {total_processed}")
    print(f"Total files that required padding:  {total_padded}")
    print(f"Total files skipped (missing/error): {total_skipped}")
    print("="*40)


if __name__ == '__main__':
    process_videos()
