import json
from pathlib import Path


models = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


print()
print(
    f"{'Model':<22}"
    f"{'Accuracy':>10}"
    f"{'Precision':>11}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'ROC-AUC':>11}"
    f"{'PR-AUC':>10}"
)

print("-" * 84)


for model in models:

    file_path = (
        Path("reports")
        / model
        / "metrics.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    print(
        f"{model:<22}"
        f"{metrics['accuracy']:>10.4f}"
        f"{metrics['precision']:>11.4f}"
        f"{metrics['recall']:>10.4f}"
        f"{metrics['f1']:>10.4f}"
        f"{metrics['roc_auc']:>11.4f}"
        f"{metrics['pr_auc']:>10.4f}"
    )