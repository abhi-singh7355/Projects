
from flask import Flask, jsonify, request

from src.logging_config import get_logger
from src.model_version import MODEL_VERSION


app = Flask(__name__)

logger = get_logger(
    "api",
    "api.log",
)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict loan default."""

    logger.info("Prediction request received")

    try:
        data = request.get_json()

        if not data:
            logger.warning("Empty prediction request")
            return jsonify({
                "error": "Request body cannot be empty"
            }), 400

        logger.info(
            "Prediction input received: %s",
            data,
        )

        # Temporary prediction for API testing.
        # Replace this with the trained model prediction.
        prediction = 0
        probability = 0.12

        logger.info(
            "Prediction completed successfully"
        )

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "model_version": MODEL_VERSION,
        }), 200

    except Exception:
        logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """API health check."""

    return jsonify({
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
