# -----------------------------------------------------------------------------------------------
# Logistic Regression Testing Code
# -----------------------------------------------------------------------------------------------
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

csv_path = 'pre_output_old/test_split.csv'

def test_logit():
    # 1. 喚醒你儲存的模型
    loaded_model = joblib.load('pre_output_old/weights.pkl')

    test_df = pd.read_csv(csv_path)
    features = ['temporal_score', 'spatial_score']
    
    X_test = test_df[features]
    y_test = test_df['binary_label']
    
    y_scores = loaded_model.predict_proba(X_test)[:, 0]
    test_df['ours'] = y_scores
    test_df.to_csv(csv_path, index=False)
    print("✅ logit測試完成，結果已儲存到 'pre_output_old/test_results.csv'")

def test_nn():
    # 1. 喚醒你儲存的模型
    model = joblib.load('pre_output_old/nn_weights.pkl')
    scaler = joblib.load('pre_output_old/scaler.pkl')

    test_df = pd.read_csv(csv_path)
    features = ['temporal_score', 'spatial_score']
    
    X_test = test_df[features]
    y_test = test_df['binary_label']
    
    X_test_scaled = scaler.transform(X_test)
    
    y_scores = model.predict_proba(X_test_scaled)[:, 0]
    test_df['ours'] = y_scores
    test_df.to_csv(csv_path, index=False)
    print("✅ neural network測試完成，結果已儲存到 'pre_output_old/test_results_nn.csv'")

def test_rf():
    # 1. 現在只要喚醒模型就好了，不需要 scaler 了
    model = joblib.load('pre_output_old/rf_weights.pkl')

    test_df = pd.read_csv(csv_path)
    features = ['temporal_score', 'spatial_score']
    
    X_test = test_df[features]
    y_test = test_df['binary_label']
    
    y_scores = model.predict_proba(X_test)[:, 0]
    test_df['ours'] = y_scores
    test_df.to_csv(csv_path, index=False)
    print("✅ random forest測試完成，結果已儲存到 'pre_output_old_old/test_results_rf.csv'")
def test_gnb():
    # 1. 載入 GaussianNB 模型
    model = joblib.load('pre_output_old/gnb_weights.pkl')

    test_df = pd.read_csv(csv_path)
    features = ['temporal_score', 'spatial_score']
    
    X_test = test_df[features]
    y_test = test_df['binary_label']

    # 2. 找出 Fake 類別在 predict_proba 裡面的 index
    fake_index = list(model.classes_).index('Fake')

    # 3. 取出 Fake 的機率分數
    y_scores = model.predict_proba(X_test)[:, fake_index]

    # 4. 存到 ours 欄位
    test_df['ours'] = y_scores
    test_df.to_csv(csv_path, index=False)

    print("✅ Gaussian Naive Bayes 測試完成，結果已儲存到 test_split.csv")
if __name__ == "__main__":
    test_logit()
    test_nn()
    test_rf()
    test_gnb()
