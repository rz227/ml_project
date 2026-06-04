import joblib
import pandas as pd

csv_path = 'pre_output_old/test_split.csv'


def get_fake_probability(model, X):
    """
    取得 Fake 類別的機率。
    這樣比直接寫 [:, 0] 安全，避免類別順序改變。
    """
    fake_index = list(model.classes_).index('Fake')
    return model.predict_proba(X)[:, fake_index]


def main():
    test_df = pd.read_csv(csv_path)
    features = ['temporal_score', 'spatial_score']
    X_test = test_df[features]

    # 1. Logistic Regression
    logit_model = joblib.load('pre_output_old/weights.pkl')
    test_df['ours_logit'] = get_fake_probability(logit_model, X_test)

    # 2. Neural Network
    nn_model = joblib.load('pre_output_old/nn_weights.pkl')
    scaler = joblib.load('pre_output_old/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    test_df['ours_nn'] = get_fake_probability(nn_model, X_test_scaled)

    # 3. Random Forest
    rf_model = joblib.load('pre_output_old/rf_weights.pkl')
    test_df['ours_rf'] = get_fake_probability(rf_model, X_test)

    # 4. Gaussian Naive Bayes
    gnb_model = joblib.load('pre_output_old/gnb_weights.pkl')
    test_df['ours_gnb'] = get_fake_probability(gnb_model, X_test)

    # 存回同一個 CSV
    test_df.to_csv(csv_path, index=False)

    print("✅ 四個模型都已完成測試")
    print("已新增欄位：ours_logit, ours_nn, ours_rf, ours_gnb")
    print(f"結果已寫回：{csv_path}")


if __name__ == "__main__":
    main()