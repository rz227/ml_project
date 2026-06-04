import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report
)

CSV_PATH = "pre_output_old/test_split.csv"
THRESHOLD = 0.5


def evaluate_score(score_col):
    df = pd.read_csv(CSV_PATH)

    label_mapping = {"Fake": 1, "Real": 0}
    y_true = df["binary_label"].map(label_mapping).astype(int)

    if score_col not in df.columns:
        print(f"❌ 找不到欄位: {score_col}")
        print("目前欄位:", list(df.columns))
        return

    temp = pd.DataFrame({
        "y_true": y_true,
        "score": pd.to_numeric(df[score_col], errors="coerce")
    }).dropna()

    if len(temp) == 0:
        print(f"❌ {score_col} 沒有可用資料")
        return

    y_true_clean = temp["y_true"]
    y_scores = temp["score"]

    ap = average_precision_score(y_true_clean, y_scores)
    auc = roc_auc_score(y_true_clean, y_scores)

    y_pred = (y_scores >= THRESHOLD).astype(int)

    acc = accuracy_score(y_true_clean, y_pred)
    pre = precision_score(y_true_clean, y_pred, zero_division=0)
    rec = recall_score(y_true_clean, y_pred, zero_division=0)
    f1 = f1_score(y_true_clean, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(f"📊 {score_col} 評估結果")
    print("=" * 60)
    print(f"有效資料筆數: {len(temp)}")
    print(f"AP:        {ap:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("-" * 60)
    print(classification_report(
        y_true_clean,
        y_pred,
        target_names=["Real", "Fake"],
        zero_division=0
    ))


if __name__ == "__main__":
    # 單一模型
    evaluate_score("temporal_score")
    evaluate_score("spatial_score")

    # 四種融合模型
    evaluate_score("ours_logit")
    evaluate_score("ours_nn")
    evaluate_score("ours_rf")
    evaluate_score("ours_gnb")