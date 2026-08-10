import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and save metrics and plots."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }

    report_dir = Path("reports") / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)

    plt.title(f"{model_name} - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(report_dir / "confusion_matrix.png")

    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, probabilities)

    plt.title(f"{model_name} - ROC Curve")

    plt.tight_layout()

    plt.savefig(report_dir / "roc_curve.png")

    plt.close()

    # Precision Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, probabilities)

    plt.title(f"{model_name} - Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(report_dir / "pr_curve.png")

    plt.close()

    return metrics
