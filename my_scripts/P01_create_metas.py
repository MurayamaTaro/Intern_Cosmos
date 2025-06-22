import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- 設定 ---
# 【変更点】全カテゴリを対象にする
CATEGORIES = ['vehicle', 'sports', 'cooking']
# 【変更点】ファイル数上限を撤廃（この行をコメントアウトまたは削除）
# PROCESS_LIMIT = 100
# ベースとなるデータセットのパス
BASE_DATA_DIR = Path('datasets/Panda-70M')
# 前処理済みデータの出力先パス
OUTPUT_DIR = Path('datasets/posttrain_panda70m')

def create_metadata_files():
    """
    Panda-70MのクリップとCSVから、後続処理用のメタデータファイル（キャプション）を生成する。
    また、新しいファイル名と元の動画ファイル名のマッピングファイルを作成する。
    """
    for category in CATEGORIES:
        print(f"--- Processing category: {category} ---")

        # --- パスの設定 ---
        clips_dir = BASE_DATA_DIR / 'clips' / category
        csv_path = BASE_DATA_DIR / 'lists' / f'{category}_candidates.csv'
        output_category_dir = OUTPUT_DIR / category
        output_meta_dir = output_category_dir / 'metas'

        # --- 出力ディレクトリの作成 ---
        output_meta_dir.mkdir(parents=True, exist_ok=True)

        # --- CSVの読み込みとキャプションの辞書作成 ---
        if not csv_path.exists():
            print(f"Warning: CSV file not found at {csv_path}. Skipping category {category}.\n")
            continue

        print(f"Loading captions from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, dtype={'videoID': str})
            df.rename(columns=lambda x: x.strip(), inplace=True)

            if 'videoID' not in df.columns or 'caption' not in df.columns:
                print(f"Error: CSV file {csv_path} must contain 'videoID' and 'caption' columns.")
                print(f"Available columns: {df.columns.tolist()}")
                continue

            df['videoID'] = df['videoID'].str.strip()
            caption_map = df.set_index('videoID')['caption'].to_dict()
            print(f"Successfully loaded {len(caption_map)} captions.")

        except Exception as e:
            print(f"Error reading or processing CSV file {csv_path}: {e}")
            continue

        # --- MP4ファイルのリストを取得 ---
        print(f"Scanning for .mp4 files in {clips_dir}...")
        mp4_files = sorted(list(clips_dir.glob('*.mp4')))

        if not mp4_files:
            print(f"Warning: No .mp4 files found in {clips_dir}. Skipping category {category}.\n")
            continue

        print(f"Found {len(mp4_files)} .mp4 files.")

        # --- メタデータファイルとマッピング情報の生成 ---
        processed_count = 0
        filename_mapping = []
        for video_path in tqdm(mp4_files, desc=f"Generating metas for {category}"):
            # 【変更点】ファイル数上限のチェックを削除
            # if processed_count >= PROCESS_LIMIT:
            #     print(f"\nReached process limit of {PROCESS_LIMIT} for {category}.")
            #     break

            try:
                video_id_for_lookup = video_path.stem.rsplit('_', 1)[0]
            except IndexError:
                continue

            if video_id_for_lookup in caption_map:
                caption = caption_map[video_id_for_lookup]

                new_base_name = f"{category}_{processed_count + 1:05d}"
                output_txt_path = output_meta_dir / f"{new_base_name}.txt"

                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(str(caption))

                filename_mapping.append({
                    'new_base_name': new_base_name,
                    'original_video_filename': video_path.name
                })

                processed_count += 1

        print(f"Successfully generated {processed_count} metadata files for {category} in {output_meta_dir}")

        # --- マッピングファイルを保存 ---
        if filename_mapping:
            mapping_df = pd.DataFrame(filename_mapping)
            mapping_csv_path = output_category_dir / 'mapping.csv'
            mapping_df.to_csv(mapping_csv_path, index=False)
            print(f"Saved filename mapping to {mapping_csv_path}\n")
        else:
            print("No matching files were found, so no mapping file was created.\n")

if __name__ == '__main__':
    create_metadata_files()
