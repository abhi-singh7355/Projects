from pathlib import Path

MODEL_VERSION = "v1.0.0"

MODEL_DIR = Path("models") / MODEL_VERSION
LATEST_MODEL_DIR = Path("models") / "latest"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LATEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
