from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = PROJECT_ROOT / "main" / "PreProcess" / "deutschl" / "erk"
ENCODED_DATASET_DIR = PROJECT_ROOT / "main" / "Data" / "dataset"
SINGLE_DATASET_PATH = PROJECT_ROOT / "main" / "Data" / "single_dataset"
MAPPING_PATH = PROJECT_ROOT / "main" / "mapping.json"
MODEL_PATH = PROJECT_ROOT / "main" / "model.h5"
OUTPUT_MIDI_PATH = PROJECT_ROOT / "main" / "mel.mid"

SEQUENCE_LENGTH = 64
OUTPUT_UNITS = 45

DEFAULT_ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
