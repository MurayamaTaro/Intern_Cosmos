import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import imageio
import decord  # OpenCVの代わりにdecordを使用
import numpy as np

# --- 設定 ---
CATEGORIES = ['vehicle']
PROCESS_LIMIT = 100

TARGET_FPS = 24
TARGET_FRAMES = 121
TARGET_RESOLUTION = (1280, 720)

BASE_DATA_DIR = Path('datasets/Panda-70M')
PROCESSED_DATA_DIR = Path('datasets/posttrain_panda70m')

def process_videos_for_test():
    """
    【テスト用・decord版】mapping.csvに基づき、動画を前処理する。
    堅牢な動画読み込みのため、OpenCVの代わりにdecordを使用する。
    """
    # decordのログレベルを設定して、不要な出力を抑制
    decord.bridge.set_bridge('native')

    for category in CATEGORIES:
        print(f"--- Processing category: {category} (TEST MODE: {PROCESS_LIMIT} files, using decord) ---")

        mapping_csv_path = PROCESSED_DATA_DIR / category / 'mapping.csv'
        original_clips_dir = BASE_DATA_DIR / 'clips' / category
        output_videos_dir = PROCESSED_DATA_DIR / category / 'videos'

        output_videos_dir.mkdir(parents=True, exist_ok=True)

        if not mapping_csv_path.exists():
            print(f"Warning: mapping.csv not found for {category}. Skipping.")
            continue

        mapping_df = pd.read_csv(mapping_csv_path)
        df_to_process = mapping_df.head(PROCESS_LIMIT)

        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"Processing videos for {category}"):
            new_base_name = row['new_base_name']
            original_filename = row['original_video_filename']

            source_video_path = original_clips_dir / original_filename
            output_video_path = output_videos_dir / f"{new_base_name}.mp4"

            if output_video_path.exists():
                continue
            if not source_video_path.exists():
                print(f"Warning: Source video not found: {source_video_path}. Skipping.")
                continue

            try:
                # 【修正点】OpenCVの代わりにdecordで動画を読み込む
                vr = decord.VideoReader(str(source_video_path), ctx=decord.cpu(0))

                processed_frames = []

                # 読み込むフレーム数を決定 (動画の全長と目標フレーム数のうち短い方)
                num_frames_to_read = min(len(vr), TARGET_FRAMES)

                for i in range(num_frames_to_read):
                    frame = vr[i]
                    # decordの出力をnumpy配列に変換
                    numpy_frame = frame.asnumpy()
                    # リサイズ処理 (decordはRGBで読み込むので色変換は不要)
                    resized_frame = cv2.resize(numpy_frame, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)
                    processed_frames.append(resized_frame)

                if not processed_frames:
                    print(f"Warning: Could not read any frames from {source_video_path} using decord. Skipping.")
                    continue

                num_existing_frames = len(processed_frames)
                if num_existing_frames < TARGET_FRAMES:
                    padding_count = TARGET_FRAMES - num_existing_frames
                    print(f"\n[Padding Info] Video '{original_filename}': Found {num_existing_frames} frames, padding with {padding_count} frames.")

                    last_frame = processed_frames[-1]
                    processed_frames.extend([last_frame] * padding_count)

                writer = imageio.get_writer(
                    output_video_path,
                    fps=TARGET_FPS,
                    codec='libx264',
                    quality=8
                )
                for frame in processed_frames:
                    writer.append_data(frame)
                writer.close()

            except decord.DECORDError as e:
                print(f"\ndecord Error processing video {original_filename}: {e}")
            except Exception as e:
                print(f"\nGeneral Error processing video {original_filename}: {e}")

if __name__ == '__main__':
    process_videos_for_test()
