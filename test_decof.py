# --------------------------------------------------------------------------------------
# a script to test the accuracy of DeCoF model on the test set (all fake videos)
# --------------------------------------------------------------------------------------

from util.extract import extract_frames, init_ff
from tqdm import tqdm
from util.dataset import VideoDatasetSplitter
import argparse
from pathlib import Path
import cv2
import sys
sys.path.append('/home/divc_col2/Group_3/DeCoF_backup/src')
from test2 import decof # type: ignore
import os
import random

# set up decof model's path
args = argparse.Namespace(
    config="/home/divc_col2/Group_3/DeCoF_backup/src/configs/base.json",
    session_name="DeCoF"
)
output_folder = Path("/home/divc_col2/Group_3/DeCoF_backup/datas/data/text2video_zero/test")

input_folder = Path("input2")
video_files = list(input_folder.glob("*.mp4")) 
total_videos = len(video_files)
correct_predictions = 0

print(f"🚀 開始處理 {total_videos} 部影片，並計算 DeCoF 準確率...")
for video_file in tqdm(video_files, desc="評估進度"):
    # preprocess video to frames
    extract_frames(video_path=video_file, out_root=output_folder, every=4, total=8, start=0)

    # decof model prediction
    scores1 = decof(args)
    #print(scores1)
    
    # Decof model highest score
    S_time = max(scores1)
    #print(S_time)
    prediction = 1 if S_time > 0.5 else 0 # 假設分數大於 0.5 就預測為 Fake，否則為 Real
    
    if(prediction == 1):
        correct_predictions += 1

accuracy = (correct_predictions / total_videos) * 100
print("\n" + "="*45)
print("📊 最終評估報告 (測試集: 全 Fake)")
print("="*45)
print(f" 📂 總測試影片數 : {total_videos} 部")
print(f" ✅ 成功預測 Fake : {correct_predictions} 部")
print(f" 🎯 總體準確率 (ACC): {accuracy:.2f}%")
print("="*45)