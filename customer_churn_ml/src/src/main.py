from src.data.load_data import load_raw_data
from src.data.preprocess import split_data
from src.fe.build_features import build_features
from src.models.train import train_models
from src.models.evaluate import evaluate_ensemble
from src.models.threshold import find_best_threshold
from src.utils.metrics import save_metrics, get_confusion_matrix
from src.paths import METRICS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    logger.info("Starting customer churn prediction pipeline")

    # 1. Load data
    df = load_raw_data("churn.csv")

    # 2. Feature engineering
    df = build_features(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Train models
    models = train_models(X_train, y_train)

    # 5. Ensemble evaluation
    results = evaluate_ensemble(models, X_test, y_test)

    # 6. Threshold optimization
    best_threshold, best_f1 = find_best_threshold(
        y_test,
        results["y_prob"]
    )

    # 7. Persist metrics
    metrics = {
        "roc_auc": results["roc_auc"],
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "classification_report": results["classification_report"],
        "confusion_matrix": get_confusion_matrix(
            y_test,
            results["y_pred"]
        )
    }

    save_metrics(
        metrics,
        METRICS_DIR / "evaluation_metrics.json"
    )

    logger.info("Pipeline finished successfully")
    logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"Best Threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    run_pipeline()

