# ---------------------------------------------------------------------------------------------------
# This program preprocess image & return decof prediction results & coresponding video frames' paths
# ---------------------------------------------------------------------------------------------------

from util.extract import extract_frames
from util.dataset import VideoDatasetSplitter
import argparse
from pathlib import Path
import sys
sys.path.append('/home/divc_col2/Group_3/DeCoF_backup/src')
from test2 import decof # type: ignore
import os
import random
from tqdm import tqdm
import shutil

# set up decof model's path
args = argparse.Namespace(
    config="/home/divc_col2/Group_3/DeCoF_backup/src/configs/base.json",
    session_name="DeCoF"
)
output_folder = Path("/home/divc_col2/Group_3/DeCoF_backup/datas/data/text2video_zero/test")

def process_dataframe(df, split_name):
    print(f"\n🚀 [階段 2] 正在處理 {split_name} 集 (共 {len(df)} 部影片)...")
    
    # clear old data output
    split_folder_path = Path("./pre_output_old") / f"{split_name}_img_list"
    if split_folder_path.exists():
        shutil.rmtree(split_folder_path) 
    split_folder_path.mkdir(parents=True, exist_ok=True)  
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} 進度"):
        vid_path = Path(row['video_path'])
        # [步驟 A]：抽幀 (8 幀)
        extract_frames(video_path=vid_path, out_root=output_folder, every=4, total=8, start=0)
        
        
        # [步驟 B]：將圖片餵給 DeCoF 算出 Temporal 分數
        scores1 = decof(args)
        
        # Decof model highest score and index
        S_time = max(scores1)
        time_index = scores1.index(S_time)
        
        # Get the image from the folder with the highest score
        base_path = '/home/divc_col2/Group_3/DeCoF_backup/datas/data/text2video_zero/test'
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        folders.sort()
        target_path = os.path.join(base_path, folders[time_index])
        jpg_files = [f for f in os.listdir(target_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if len(jpg_files) > 0:
            # target image path
            random_image_name = random.choice(jpg_files)
            target_image_path = os.path.join(target_path, random_image_name)
            # print(f"Target image path: {target_image_path}")
        else:
            # print("No jpg files found in the target folder.")
            target_image_path = None


        # [步驟 C]：把算出來的分數，準確地填入表格對應的格子裡
        df.at[index, 'temporal_score'] = S_time
        
        # [步驟 D]：將最高分數圖片暫存至臨時的資料夾，並把對應的圖片路徑，準確地填入表格對應的格子裡
        if target_image_path is not None and os.path.exists(target_image_path):
            
            # 1. 決定要存進哪個資料夾
            split_folder_name = f"{split_name}_img_list" 
            safe_save_dir = Path("./pre_output_old") / split_folder_name 
            safe_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 幫圖片取一個獨一無二的新名字
            video_stem = vid_path.stem
            new_image_path = safe_save_dir / f"{video_stem}.jpg"
            
            # 3. 把圖片從隨時會被清空的暫存區，複製到我們剛剛建好的安全資料夾
            shutil.copy2(target_image_path, new_image_path)
            
            # 4. 將這個「絕對安全的路徑」寫回 DataFrame 中
            df.at[index, 'best_frame_path'] = str(new_image_path)
        else:
            # 如果這部影片剛剛沒抽到半張圖，就填入 None
            df.at[index, 'best_frame_path'] = None
        
        


def preprocess(split_name):
    # split data into train, val and test
    splitter = VideoDatasetSplitter(base_dir="./input_old", random_state=42)
    
    splitter.load_data()
    train_df, val_df, test_df = splitter.split_data(train_size=0.7, val_size=0.15, test_size=0.15)
    
    
    
    # process train, val, test dataframes and save the results into csv files
    if split_name == 'train':
        process_dataframe(train_df, "train")
    if split_name == 'val':
        process_dataframe(val_df, "val")
    if split_name == 'test':
        process_dataframe(test_df, "test")
    

    # save the split data into csv files
    splitter.save_to_csv(output_dir="./pre_output_old", split_name=split_name)