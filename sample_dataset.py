# --------------------------------------------------------------------------------------
# Template code for demonstrate the shape of the train, val, test data after splitting
# --------------------------------------------------------------------------------------

# 引入剛剛寫好的 Class
from dataset2 import VideoDatasetSplitter


def func1():
    # 1. 實例化物件 (建立工具)
    splitter = VideoDatasetSplitter(base_dir="./input", random_state=42)

    # 2. 執行讀取與切分
    splitter.load_data()
    train_data, val_data, test_data = splitter.split_data(train_size=0.7, val_size=0.15, test_size=0.15)
    splitter.print_summary()  # 印出切分後的資料統計報告
    # splitter.save_all_csv(output_dir="./pre_output")  # 將切分結果存成 CSV
    print(val_data)  # 查看切分後的 Val 資料前幾筆


def func2():
    # 1. 實例化 Splitter
    splitter = VideoDatasetSplitter(base_dir="./input")

    # 2. 定義對應你目前真實資料夾結構的字典
    my_custom_config = {
        # 真實影片
        "0_real": {"fine_label": "Real", "binary_label": "Real"},
        
        # Deepfake 影片
        "1_fake/deepfake": {"fine_label": "Deepfake", "binary_label": "Fake"},
        
        # 🌟 直接指向 aigc 底下的 sora 和 gen3 資料夾
        "1_fake/aigc/sora": {"fine_label": "Sora", "binary_label": "Fake"},
        "1_fake/aigc/Gen3": {"fine_label": "Gen3", "binary_label": "Fake"}
    }

    # 3. 讀取資料
    splitter.load_data(dataset_config=my_custom_config)

    # 4. 切分與印出報告
    train_data, val_data, test_data = splitter.split_data()
    splitter.print_summary()
    splitter.save_all_csv(output_dir="./pre_output")  # 將切分結果存成 CSV
    print(train_data)  # 查看切分後的 Train 資料前幾筆
    
if __name__ == "__main__":
    func2()