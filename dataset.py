# --------------------------------------------------------------------------------------
# create dataset class object to read video files and split into train, val, test
# --------------------------------------------------------------------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split

class VideoDatasetSplitter:
    def __init__(self, base_dir="./input", random_state=42):
        """
        初始化分割器，設定基礎路徑與亂數種子
        """
        self.base_dir = base_dir
        self.random_state = random_state
        self.valid_extensions = ('.mp4', '.avi', '.mov')
        
        # 用來儲存各個階段的 DataFrame
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def _load_folder(self, folder_path, fine_label, binary_label, video_data):
        """(內部方法) 讀取單一資料夾"""
        if not os.path.exists(folder_path):
            print(f"⚠️ 找不到資料夾: {folder_path}")
            return
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(self.valid_extensions):
                full_path = os.path.join(folder_path, filename)
                video_data.append({
                    'video_path': full_path,
                    'fine_label': fine_label,
                    'binary_label': binary_label,
                    'best_frame_path': None,
                    'temporal_score': None,
                    'spatial_score': None
                })

    def load_data(self):
        """讀取所有指定結構的影片資料並轉為 DataFrame"""
        real_dir = os.path.join(self.base_dir, "0_real")
        deepfake_dir = os.path.join(self.base_dir, "1_fake", "deepfake")
        aigc_dir = os.path.join(self.base_dir, "1_fake", "aigc")

        video_data = []
        self._load_folder(real_dir, fine_label="Real", binary_label="Real", video_data=video_data)
        self._load_folder(deepfake_dir, fine_label="Deepfake", binary_label="Fake", video_data=video_data)
        self._load_folder(aigc_dir, fine_label="AIGC", binary_label="Fake", video_data=video_data)

        self.df = pd.DataFrame(video_data)
        print(f"✅ 成功讀取 {len(self.df)} 支影片！\n")
        return self.df

    def split_data(self, train_size=0.70, val_size=0.15, test_size=0.15):
        """進行分層抽樣切分 (Stratified Split)"""
        if self.df is None or len(self.df) == 0:
            raise ValueError("資料為空！請先執行 load_data()")

        # 計算第一刀的比例 (1 - train_size)
        temp_size = val_size + test_size
        
        # 第一刀：切出 Train 與 Temp
        self.train_df, temp_df = train_test_split(
            self.df, 
            test_size=temp_size, 
            random_state=self.random_state, 
            stratify=self.df['fine_label']
        )

        # 計算第二刀的比例 (test 佔 temp 的多少)
        relative_test_size = test_size / temp_size

        # 第二刀：將 Temp 切成 Val 與 Test
        self.val_df, self.test_df = train_test_split(
            temp_df, 
            test_size=relative_test_size, 
            random_state=self.random_state, 
            stratify=temp_df['fine_label']
        )
        return self.train_df, self.val_df, self.test_df

    def print_summary(self):
        """印出資料集切分前後的詳細統計報告"""
        if self.train_df is None:
            print("請先執行 split_data()！")
            return

        print("=== 切分前的原始資料數量 ===")
        print(self.df['fine_label'].value_counts())
        print("==========================\n")
        
        print(f"📊 切分結果 (總數 {len(self.df)} 筆):")
        print("-" * 40)
        for name, dataset in zip(["Train", "Val", "Test"], [self.train_df, self.val_df, self.test_df]):
            print(f"{name} 集 ({len(dataset)} 筆):")
            print(dataset['fine_label'].value_counts(normalize=True).round(3))
            print("-" * 40)

    def save_to_csv(self, output_dir="./", split_name="dataset"):
        """將切分好的結果存成 CSV"""
        if self.train_df is None:
            print("請先執行 split_data()！")
            return
        if(split_name == "train"):
            self.train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
        if(split_name == "val"):
            self.val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
        if(split_name == "test"):
            self.test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
            
        print("🎉 CSV 檔案已成功生成！")
    def save_all_csv(self, output_dir="./"):
        """將切分好的結果存成 CSV"""
        if self.train_df is None:
            print("請先執行 split_data()！")
            return
        
        self.train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
        self.val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
        self.test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
            
        print("🎉 CSV 檔案已成功生成！")