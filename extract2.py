# ---------------------------------------------------------------------------------------------------
# Final version of extract for the real apllication
# ---------------------------------------------------------------------------------------------------

import argparse
from pathlib import Path
import cv2
import shutil

def extract_frames(video_path: Path, out_root: Path, every: int = 4, total: int = 8, start: int = 0):
    # 執行前先清空舊的 output 資料夾
    if out_root.exists():
        #print(f"[清理] 正在刪除舊的資料夾: {out_root} ...")
        shutil.rmtree(out_root) # 連同裡面所有檔案一起刪除

    if every <= 0:
        raise ValueError("--every must be >= 1")
    if total <= 0:
        raise ValueError("--total must be >= 1")
    if start < 0:
        raise ValueError("--start must be >= 0")

    video_path = Path(video_path)
    out_root = Path(out_root)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_in_seg = 0
    seg_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ---------------------------------------------------------
            # 【全新修改點】：直接看當前畫面的「時間戳記 (毫秒)」
            # ---------------------------------------------------------
            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # 30 秒 = 30,000 毫秒。如果超過這個時間，直接結束擷取！
            if current_time_msec >= 30000:
                # print(f"已到達 30 秒限制 (目前時間: {current_time_msec} ms)，停止擷取。")
                break
            # ---------------------------------------------------------
            
            if frame_idx >= start and (frame_idx - start) % every == 0:

                if saved_in_seg == 0:
                    # 【修改點 2】：加入 :03d 讓數字變成三位數，例如 seg001, seg002
                    current_out_dir = out_root / f"{video_path.stem}_seg{seg_counter:03d}"
                    current_out_dir.mkdir(parents=True, exist_ok=True)
                    # print(f"--> Processing Segment {seg_counter}: {current_out_dir}")

                out_file = current_out_dir / f"{saved_in_seg:03d}.jpg"
                cv2.imwrite(str(out_file), frame)

                saved_in_seg += 1

                if saved_in_seg >= total:
                    seg_counter += 1
                    saved_in_seg = 0

            frame_idx += 1

        if 0 < saved_in_seg < total and current_out_dir is not None and current_out_dir.exists():
            #print(f"\n[清理] 發現不完整的 Segment {seg_counter} (僅 {saved_in_seg}/{total} 幀)，正在移除該資料夾...")
            shutil.rmtree(current_out_dir)
            seg_counter -= 1

    finally:
        cap.release()

    #print(f"\n[Done] Processed {frame_idx} frames.")
    #print(f"Total segments created: {seg_counter - 1 if saved_in_seg == 0 else seg_counter}")


def main():
    video_file = Path("/home/divc_col2/Group_3/input/tmp2.mp4")
    output_folder = Path("/home/divc_col2/Group_3/output")

    # 建立一個全新的、乾淨的 output 資料夾
    output_folder.mkdir(parents=True, exist_ok=True)

    if video_file.exists():
        extract_frames(
            video_path=video_file,
            out_root=output_folder,
            every=4,
            total=8,
            start=0
        )
    else:
        print(f"找不到影片: {video_file}，請修改 main 函數中的路徑測試。")


if __name__ == "__main__":
    main()

from glob import glob
import os
from glob import glob
import os
import sys
import numpy as np
from PIL import Image
import random
import warnings
import json
import os.path as osp



def init_ff(data_name='input'):
	dataset_path_r=os.path.join(data_name,'0_real/')

	dataset_path_f=os.path.join(data_name,'1_fake/')





	real_img_list   = sorted(glob(dataset_path_r+'*'))
	fake_img_list  = sorted(glob(dataset_path_f+'*'))





	fake_label_list = [1 for _ in range(len(fake_img_list))]
    #print(fake_img_list[0])



	real_label_list = [0 for _ in range(len(real_img_list))]

	img = real_img_list+fake_img_list
	label = real_label_list+fake_label_list

	return img,label