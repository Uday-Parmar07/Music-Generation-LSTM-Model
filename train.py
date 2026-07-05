import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components.preprocess import run_preprocessing_pipeline
from src.components.train import save_history, train_model


if __name__ == "__main__":
    # Fast default for quick iterations.
    _, history = train_model(preset="fastest")
    save_history(history, PROJECT_ROOT / "main" / "training_history.json")
    print(f"Final eval: {history['final_eval']}")
    print("Training complete. Model saved to main/model.h5")
