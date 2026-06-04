# -----------------------------------------------------------------------------------------------
# Training Code for all methods
# -----------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
# Gaussian Naive Bayes
def train_gnb():
    # 1. 讀取資料
    df = pd.read_csv('pre_output_old/val_split.csv') 

    # 2. 使用兩個融合特徵
    features = ['temporal_score', 'spatial_score'] 
    X = df[features] 
    y = df['binary_label']  

    # 3. 建立 Gaussian Naive Bayes 模型
    gnb_model = GaussianNB()

    # 4. 訓練模型
    gnb_model.fit(X, y)

    # 5. 訓練集上測試效果
    y_pred = gnb_model.predict(X)

    print("-" * 30)
    print("📘 Gaussian Naive Bayes 訓練完成！")
    print(f"整體準確率: {accuracy_score(y, y_pred) * 100:.2f}%\n")

    print("詳細分類報告:")
    print(classification_report(y, y_pred))
    print("-" * 30)

    # 6. 儲存模型
    model_filename = 'pre_output_old/gnb_weights.pkl'
    joblib.dump(gnb_model, model_filename)

    print(f"💾 GaussianNB 模型已經成功儲存在: {model_filename}")
    print("-" * 30)
# logistic regression
def train_logit():
    # 1. 讀取你千辛萬苦跑出來的真實表格
    df = pd.read_csv('pre_output_old/val_split.csv') 

    # 2. 只有這兩個你真正算出來的分數特徵
    features = ['temporal_score', 'spatial_score'] 
    X = df[features] 

    y = df['binary_label']  

    # 4. 訓練模型
    lr_model = LogisticRegression()
    lr_model.fit(X, y)

    # 5. 看看模型考幾分
    y_pred = lr_model.predict(X)
    
    print("-" * 30)
    print("🚀 邏輯回歸訓練完成！")
    print(f"整體準確率: {accuracy_score(y, y_pred) * 100:.2f}%\n")
    
    # 印出詳細成績單，你會看到它直接用 Real 和 Fake 來幫你算分！
    print("詳細分類報告 (看它抓 Fake 的能力好不好):")
    print(classification_report(y, y_pred))
    print("-" * 30)

    # 6. 將訓練好的超強大腦（模型）存檔
    model_filename = 'pre_output_old/weights.pkl' 
    joblib.dump(lr_model, model_filename)
    print(f"💾 模型已經成功儲存在: {model_filename}")
    print("-" * 30)

# Neural Network
def train_nn():
    # 1. 讀取你千辛萬苦跑出來的真實表格
    df = pd.read_csv('pre_output_old/val_split.csv') 

    # 2. 只有這兩個你真正算出來的分數特徵
    features = ['temporal_score', 'spatial_score'] 
    X = df[features] 
    y = df['binary_label']  

    # --- 🌟 新增步驟：特徵縮放 (神經網路必做！) ---
    # 這會把你的分數轉換成平均值為 0、標準差為 1 的範圍，讓神經網路更好消化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # -----------------------------------------------

    # 4. 訓練模型 (替換成神經網路)
    # hidden_layer_sizes=(8,) 代表：建立「1個隱藏層」，裡面有「8個神經元」。
    # 加上原本的輸入層與輸出層，這就是你所說的兩層神經網路架構！
    nn_model = MLPClassifier(
        hidden_layer_sizes=(8,), # 你可以隨時修改這裡的數字來增加腦容量，例如 (16,)  
        activation='relu',       # 啟動函數，relu 是目前最好用的標準配置
        max_iter=1000,           # 最大訓練次數，神經網路需要多跑幾次才會收斂
        random_state=42
    )
    
    # 注意：這裡要餵進去的是「縮放過」的資料
    nn_model.fit(X_scaled, y)

    # 5. 看看模型考幾分
    # 注意：考試的時候，特徵也必須經過同樣的縮放！
    y_pred = nn_model.predict(X_scaled)
    
    print("-" * 30)
    print("🚀 神經網路訓練完成！")
    print(f"整體準確率: {accuracy_score(y, y_pred) * 100:.2f}%\n")
    
    print("詳細分類報告:")
    print(classification_report(y, y_pred))
    print("-" * 30)

    # 6. 將訓練好的超強大腦（模型）以及「縮放器」一起存檔
    model_filename = 'pre_output_old/nn_weights.pkl' 
    scaler_filename = 'pre_output_old/scaler.pkl' # ⚠️ 縮放器也必須存下來！
    
    joblib.dump(nn_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"💾 神經網路模型已經成功儲存在: {model_filename}")
    print(f"💾 特徵縮放器已經成功儲存在: {scaler_filename}")
    print("-" * 30)


def train_nn2():
    # 1. 讀取你「所有」的資料
    # 請確保這份 CSV 包含了你想用來訓練的所有數據
    df = pd.read_csv('pre_output_old/val_split.csv') 
    
    features = ['temporal_score', 'spatial_score']
    X = df[features]
    y = df['binary_label'] # 假設 0 是 Fake, 1 是 Real

    # =========================================================
    # 🌟 關鍵步驟 1：特徵縮放 (Standardization)
    # 這次我們不切資料了，直接對「全部的 X」進行 fit_transform
    # =========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 儲存 Scaler，讓推論程式可以無縫接軌使用
    joblib.dump(scaler, 'pre_output_old/scaler.pkl')
    print("✅ 縮放器已儲存至 scaler.pkl")

    # =========================================================
    # 🌟 關鍵步驟 2：建立神經網路
    # hidden_layer_sizes=(16,): 代表 1 層隱藏層 (16顆神經元)。
    # 若想挑戰 2 層隱藏層，可改成 (16, 8) 試試看！
    # =========================================================
    nn_model = MLPClassifier(
        hidden_layer_sizes=(16,), 
        activation='relu',        
        solver='adam',            
        max_iter=1000,            
        random_state=42
    )
    
    print("🧠 神經網路開始訓練 (使用 100% 的資料)...")
    # 直接把所有資料餵進去！
    nn_model.fit(X_scaled, y)
    
    # 備註：因為沒有驗證集了，所以這裡不會印出驗證集的準確率
    print("🎉 訓練完成！")

    # =========================================================
    # 🌟 關鍵步驟 3：儲存最終模型
    # =========================================================
    joblib.dump(nn_model, 'pre_output_old/nn_weights.pkl')
    print("✅ 神經網路模型已儲存至 nn_weights.pkl")

# Random Forest
def train_rf():
    # 1. 讀取你千辛萬苦跑出來的真實表格
    df = pd.read_csv('pre_output_old/val_split.csv') 

    # 2. 只有這兩個你真正算出來的分數特徵
    features = ['temporal_score', 'spatial_score'] 
    X = df[features] 
    y = df['binary_label']  

    # 4. 建立與訓練模型 (隨機森林)
    rf_model = RandomForestClassifier(
        n_estimators=100,  # 建立 100 棵決策樹來共同投票
        max_depth=5,       # 限制樹的深度，防止死背 (過擬合)
        random_state=42
    )
    
    # 直接把 100% 的 X 餵進去
    rf_model.fit(X, y)

    # 5. 看看模型考幾分
    # 同樣地，直接用100%的 X 來考試
    y_pred = rf_model.predict(X)
    
    print("-" * 30)
    print("🌲 隨機森林模型訓練完成！")
    print(f"整體準確率: {accuracy_score(y, y_pred) * 100:.2f}%\n")
    
    print("詳細分類報告:")
    print(classification_report(y, y_pred))
    print("-" * 30)

    # 6. 將訓練好的森林存檔 (現在只需要存一個檔案了！)
    model_filename = 'pre_output_old/rf_weights.pkl' 
    
    joblib.dump(rf_model, model_filename)
    
    print(f"💾 隨機森林模型已經成功儲存在: {model_filename}")
    # (之前的 scaler_filename 已經不需要存了)
    print("-" * 30)
    
def temp():
    # 1. 讀取真實表格
    try:
        df = pd.read_csv('pre_output_old_copy/val_split.csv') 
    except FileNotFoundError:
        print("❌ 找不到 CSV 檔案，請確認路徑是否正確。")
        return

    # 2. 特徵與標籤
    features = ['temporal_score', 'spatial_score'] 
    X = df[features] 
    y = df['binary_label']  

    # 3. 建立與訓練模型 (隨機森林)
    rf_model = RandomForestClassifier(
        n_estimators=100,  # 100 棵樹
        max_depth=5,       # 限制深度
        random_state=42
    )
    
    rf_model.fit(X, y)

    # 3. 🌟 核心：直接計算數據 (不用存檔)
    # Parameters: 統計所有樹的總節點數
    total_params = sum(tree.tree_.node_count for tree in rf_model.estimators_)

    # Complexity: 隨機森林主要運算為比較，最大運算量為 樹量 * 深度
    total_ops = rf_model.n_estimators * rf_model.max_depth

    # 4. 印出結果
    print("\n" + "="*40)
    print("📊 隨機森林複雜度分析結果")
    print("-" * 40)
    print(f"🔹 Parameters (總節點數): {total_params}")
    print(f"🔹 運算量 (最大比較次數): {total_ops} 次")
    print("="*40)
    print("\n🎉 數據拿到了！你可以直接把這兩個數字填進報告，不需要儲存 .pkl 檔。")

if __name__ == "__main__":
    train_logit()
    train_nn2()
    train_rf()
    train_gnb()
    #temp()