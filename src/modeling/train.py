import json
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import create_features
from src.logging_config import get_logger
from src.model_version import (
    LATEST_MODEL_DIR,
    MODEL_DIR,
    MODEL_VERSION,
)
from src.reports import evaluate_model

logger = get_logger("training", "training.log")


DATA_PATH = "data/raw/Loan_Default.csv"

TARGET = "Status"


def build_preprocessor(X):

    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    logger.info("Numerical features: %s", len(numeric_columns))

    logger.info("Categorical features: %s", len(categorical_columns))

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor


def train_models():

    logger.info("====================================")

    logger.info("Model training started")

    logger.info("Model version: %s", MODEL_VERSION)

    try:
        # --------------------------------
        # Load dataset
        # --------------------------------

        logger.info("Loading dataset")

        df = pd.read_csv(DATA_PATH)

        logger.info("Dataset shape: %s", df.shape)

        # --------------------------------
        # Feature engineering
        # --------------------------------

        df = create_features(df)

        # --------------------------------
        # Target
        # --------------------------------

        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found")

        df = df.dropna(subset=[TARGET])

        y = df[TARGET]

        X = df.drop(columns=[TARGET])

        # Convert target to integer
        y = y.astype(int)

        logger.info("Target distribution:\n%s", y.value_counts())

        # --------------------------------
        # Drop ID
        # --------------------------------

        if "ID" in X.columns:
            X = X.drop(columns=["ID"])

            logger.info("Dropped ID column")

        # --------------------------------
        # Train test split
        # --------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        logger.info("Training rows: %s", len(X_train))

        logger.info("Testing rows: %s", len(X_test))

        # --------------------------------
        # Preprocessor
        # --------------------------------

        preprocessor = build_preprocessor(X_train)

        # --------------------------------
        # Models
        # --------------------------------

        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        }

        all_metrics = {}

        # --------------------------------
        # Train each model
        # --------------------------------

        for model_name, classifier in models.items():
            logger.info("Training %s", model_name)

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        classifier,
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)

            logger.info("%s training completed", model_name)

            # --------------------------------
            # Evaluation
            # --------------------------------

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test,
                model_name,
            )

            all_metrics[model_name] = metrics

            logger.info("%s metrics: %s", model_name, metrics)

            # --------------------------------
            # Save model
            # --------------------------------

            model_path = MODEL_DIR / f"{model_name}.joblib"

            joblib.dump(pipeline, model_path)

            logger.info("Saved model: %s", model_path)

            # Save latest model
            latest_model_path = LATEST_MODEL_DIR / f"{model_name}.joblib"

            shutil.copy2(model_path, latest_model_path)

        # --------------------------------
        # Best model
        # --------------------------------

        best_model = max(all_metrics, key=lambda name: all_metrics[name]["roc_auc"])

        # --------------------------------
        # Metadata
        # --------------------------------

        metadata = {
            "model_version": MODEL_VERSION,
            "target": TARGET,
            "best_model": best_model,
            "models": all_metrics,
            "leakage_removed": [
                "rate_of_interest",
                "Interest_rate_spread",
                "Upfront_charges",
            ],
            "features_created": [
                "property_value_isna",
                "dtir1_isna",
                "DTI_x_LTV",
            ],
        }

        metadata_path = MODEL_DIR / "metadata.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=4)

        logger.info("Metadata saved: %s", metadata_path)

        # Save metadata to latest
        latest_metadata_path = LATEST_MODEL_DIR / "metadata.json"

        shutil.copy2(metadata_path, latest_metadata_path)

        logger.info("Best model: %s", best_model)

        logger.info("All models trained successfully")

        return all_metrics

    except Exception:
        logger.exception("Model training failed")

        raise


if __name__ == "__main__":
    train_models()
