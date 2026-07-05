"""Backward-compatible wrapper around src.components.preprocess."""

from src.components.config import (
    DEFAULT_ACCEPTABLE_DURATIONS as ACCEPTABLE_DURATIONS,
    DEFAULT_DATASET_PATH,
    ENCODED_DATASET_DIR as SAVE_DIR,
    MAPPING_PATH,
    SEQUENCE_LENGTH,
    SINGLE_DATASET_PATH as SINGLE_DATASET,
)
from src.components.preprocess import (
    create_mapping,
    create_single_file_dataset,
    encode_song,
    generate_training_sequences,
    has_acceptable_durations,
    load,
    load_data,
    preprocess,
    run_preprocessing_pipeline,
    songs_to_int,
    transpose,
)

KERN_DATASET_PATH = DEFAULT_DATASET_PATH


def main():
    run_preprocessing_pipeline(
        dataset_path=KERN_DATASET_PATH,
        save_dir=SAVE_DIR,
        single_dataset_path=SINGLE_DATASET,
        mapping_path=MAPPING_PATH,
        sequence_length=SEQUENCE_LENGTH,
    )


if __name__ == "__main__":
    main()