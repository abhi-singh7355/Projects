import pandas as pd

from src.logging_config import get_logger

logger = get_logger("dataset", "pipeline.log")


def load_data(path):
    """Load the raw dataset from CSV."""

    logger.info("Starting dataset loading")

    try:
        df = pd.read_csv(path)

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)

        return df

    except Exception as exc:
        logger.error("Dataset loading failed: %s", exc)
        raise


if __name__ == "__main__":
    data_path = "data/raw/Loan_Default.csv"

    df = load_data(data_path)

    print(df.head())
