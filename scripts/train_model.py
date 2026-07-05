import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components.preprocess import run_preprocessing_pipeline
from src.components.train import save_history, train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the music model")
    parser.add_argument(
        "--preset",
        choices=["fastest", "balanced", "quality"],
        default="fastest",
        help="Training speed/quality preset.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run dataset preprocessing before training.",
    )
    args = parser.parse_args()

    if args.preprocess:
        preprocessing_summary = run_preprocessing_pipeline()
        print(f"Preprocessing: {preprocessing_summary}")

    _, history = train_model(preset=args.preset)
    save_history(history, PROJECT_ROOT / "main" / "training_history.json")
    print(f"Final eval: {history['final_eval']}")
    print("Training complete. Model saved to main/model.h5")
