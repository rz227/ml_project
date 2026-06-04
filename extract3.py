# ---------------------------------------------------------------------------------------------------
# Extract program for training decof
# ---------------------------------------------------------------------------------------------------

import pandas as pd
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_single_segment(video_path: Path, current_out_dir: Path, every: int = 4, total: int = 8, start: int = 0):
    """
    (精簡版) 只從影片中抽取「一段」固定數量的畫面，並存入指定的資料夾。
    """
    # 如果資料夾已存在，先清空它 (避免殘留舊圖片)
    if current_out_dir.exists():
        shutil.rmtree(current_out_dir)
    current_out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ 無法開啟影片: {video_path}")
        return False

    frame_idx = 0
    saved_in_seg = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 達到起始點，且符合間隔條件
            if frame_idx >= start and (frame_idx - start) % every == 0:
                out_file = current_out_dir / f"{saved_in_seg:03d}.jpg"
                cv2.imwrite(str(out_file), frame)
                saved_in_seg += 1

                # 只要取滿 total (8) 張，直接跳出迴圈結束！
                if saved_in_seg >= total:
                    break

            frame_idx += 1

        # 如果影片太短，連一段 (8張) 都湊不齊，屬於無效資料，將空資料夾刪除
        if saved_in_seg < total:
            # print(f"影片太短，跳過: {video_path.name}")
            shutil.rmtree(current_out_dir)
            return False

        return True
    finally:
        cap.release()

def process_dataset(phase: str, base_output_dir: str):
    """
    核心邏輯：讀取 CSV，根據 phase 與 binary_label 分配資料夾並抽取畫面。
    """
    if phase not in ["train", "val"]:
        raise ValueError("參數錯誤：phase 必須是 'train' 或 'val'")

    csv_path = Path(f"/home/divc_col2/Group_3/pre_output_old/{phase}_split.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 檔案: {csv_path}")

    # 讀取 CSV
    df = pd.read_csv(csv_path)
    print(f"🚀 開始處理 {phase} 階段，共 {len(df)} 部影片...")

    # 使用 tqdm 顯示進度條
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_path = Path(row['video_path'])
        binary_label = row['binary_label']
        finlabel = row['fine_label']

        # 根據標籤決定子資料夾名稱
        if binary_label == "Real":
            label_folder = "0_real"
        elif binary_label == "Fake":
            label_folder = "1_fake"
        else:
            continue  # 若有例外標籤則直接跳過

        # 組合最終儲存路徑，例如： ./dataset/train/0_real/影片檔名_seg000
        target_dir = Path(base_output_dir) / phase / label_folder / f"{finlabel}_{video_path.stem}"

        # 執行抽取 (只取一段)
        extract_single_segment(
            video_path=video_path,
            current_out_dir=target_dir,
            every=4,
            total=8,
            start=0
        )

if __name__ == "__main__":
    # 設定你要輸出資料夾的「根目錄」
    # (程式會自動在這個目錄下建立 train/ 和 val/ 以及裡面的 0_real, 1_fake)
    OUTPUT_ROOT = "/home/divc_col2/Group_3/DeCoF_backup/datas/data/text2video_zero" 

    # 執行 Train 集處理
    process_dataset(phase="train", base_output_dir=OUTPUT_ROOT)

    # 執行 Val 集處理
    process_dataset(phase="val", base_output_dir=OUTPUT_ROOT)
    
    print("🎉 所有影片處理完畢！")