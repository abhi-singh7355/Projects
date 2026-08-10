from src.logging_config import get_logger

logger = get_logger(
    "features",
    "pipeline.log",
)


LEAKAGE_COLUMNS = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
]


def create_features(df):
    """Create production features for loan default prediction."""

    logger.info("Feature engineering started")

    try:
        df = df.copy()

        logger.info(
            "Input dataset shape: %s",
            df.shape,
        )

        # -----------------------------
        # Missingness indicators
        # -----------------------------

        if "property_value" in df.columns:
            df["property_value_isna"] = df["property_value"].isna().astype(int)

            logger.info("Created property_value_isna")

        if "dtir1" in df.columns:
            df["dtir1_isna"] = df["dtir1"].isna().astype(int)

            logger.info("Created dtir1_isna")

        # -----------------------------
        # DTI x LTV
        # -----------------------------

        if "dtir1" in df.columns and "LTV" in df.columns:
            df["DTI_x_LTV"] = df["dtir1"] * df["LTV"]

            logger.info("Created DTI_x_LTV")

        # -----------------------------
        # Remove leakage
        # -----------------------------

        existing_leakage = [
            column for column in LEAKAGE_COLUMNS if column in df.columns
        ]

        if existing_leakage:
            df = df.drop(columns=existing_leakage)

            logger.warning(
                "Removed leakage columns: %s",
                existing_leakage,
            )

        # -----------------------------
        # Missing values logging
        # -----------------------------

        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            logger.warning(
                "Missing values remaining: %s",
                missing_count,
            )

        logger.info(
            "Feature engineering completed. Output shape: %s",
            df.shape,
        )

        return df

    except Exception:
        logger.exception("Feature engineering failed")
        raise
