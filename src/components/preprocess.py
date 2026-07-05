import json
from pathlib import Path

import music21 as m21
import numpy as np

from src.components.config import (
    DEFAULT_ACCEPTABLE_DURATIONS,
    DEFAULT_DATASET_PATH,
    ENCODED_DATASET_DIR,
    MAPPING_PATH,
    SEQUENCE_LENGTH,
    SINGLE_DATASET_PATH,
)


def load_data(dataset_path):
    """Load all .krn songs from a directory tree."""
    songs = []
    dataset_path = Path(dataset_path)

    for file_path in dataset_path.rglob("*.krn"):
        songs.append(m21.converter.parse(file_path))

    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """Transpose each song to C major or A minor."""
    key = song.analyze("key")

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    else:
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    return song.transpose(interval)


def encode_song(song, time_step=0.25):
    encoded_song = []

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        else:
            symbol = "r"

        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            encoded_song.append(symbol if step == 0 else "_")

    return " ".join(map(str, encoded_song))


def load(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read()


def preprocess(
    dataset_path=DEFAULT_DATASET_PATH,
    save_dir=ENCODED_DATASET_DIR,
    acceptable_durations=DEFAULT_ACCEPTABLE_DURATIONS,
):
    """Load, filter, transpose, encode, and save songs as token text files."""
    dataset_path = Path(dataset_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    songs = load_data(dataset_path)

    saved_count = 0
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, acceptable_durations):
            continue

        song = transpose(song)
        encoded_song = encode_song(song)

        save_path = save_dir / str(i)
        with open(save_path, "w", encoding="utf-8") as fp:
            fp.write(encoded_song)
        saved_count += 1

    return {"loaded": len(songs), "saved": saved_count}


def create_single_file_dataset(
    dataset_path=ENCODED_DATASET_DIR,
    file_dataset_path=SINGLE_DATASET_PATH,
    sequence_length=SEQUENCE_LENGTH,
):
    dataset_path = Path(dataset_path)
    file_dataset_path = Path(file_dataset_path)
    file_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    for file_path in sorted(dataset_path.rglob("*")):
        if file_path.is_file():
            song = load(file_path)
            songs += song + " " + new_song_delimiter

    songs = songs[:-1]

    with open(file_dataset_path, "w", encoding="utf-8") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path=MAPPING_PATH):
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = songs.split()
    vocabulary = sorted(set(symbols))
    mappings = {symbol: i for i, symbol in enumerate(vocabulary)}

    with open(mapping_path, "w", encoding="utf-8") as fp:
        json.dump(mappings, fp, indent=4)

    return mappings


def songs_to_int(songs, mapping_path=MAPPING_PATH):
    mapping_path = Path(mapping_path)

    with open(mapping_path, "r", encoding="utf-8") as fp:
        mappings = json.load(fp)

    return [mappings[symbol] for symbol in songs.split()]


def generate_training_sequences(
    sequence_length=SEQUENCE_LENGTH,
    stride=1,
    max_sequences=None,
    one_hot_inputs=False,
    single_dataset_path=SINGLE_DATASET_PATH,
    mapping_path=MAPPING_PATH,
):
    """Create integer (or optionally one-hot) inputs and integer targets."""
    songs = load(single_dataset_path)
    int_songs = songs_to_int(songs, mapping_path)

    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(0, num_sequences, stride):
        inputs.append(int_songs[i : i + sequence_length])
        targets.append(int_songs[i + sequence_length])

        if max_sequences is not None and len(inputs) >= max_sequences:
            break

    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    if one_hot_inputs:
        # Import lazily so preprocessing-only workflows avoid loading keras.
        import keras

        vocabulary_size = len(set(int_songs))
        inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)

    return inputs, targets


def run_preprocessing_pipeline(
    dataset_path=DEFAULT_DATASET_PATH,
    save_dir=ENCODED_DATASET_DIR,
    single_dataset_path=SINGLE_DATASET_PATH,
    mapping_path=MAPPING_PATH,
    sequence_length=SEQUENCE_LENGTH,
):
    stats = preprocess(dataset_path=dataset_path, save_dir=save_dir)
    songs = create_single_file_dataset(
        dataset_path=save_dir,
        file_dataset_path=single_dataset_path,
        sequence_length=sequence_length,
    )
    mapping = create_mapping(songs=songs, mapping_path=mapping_path)

    return {
        "loaded_songs": stats["loaded"],
        "saved_songs": stats["saved"],
        "vocab_size": len(mapping),
    }
