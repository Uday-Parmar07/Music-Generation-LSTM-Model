import json

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from src.components.config import (
    MAPPING_PATH,
    MODEL_PATH,
    OUTPUT_UNITS,
    SEQUENCE_LENGTH,
)
from src.components.model import build_model
from src.components.preprocess import generate_training_sequences


TRAINING_PRESETS = {
    "fastest": {
        "epochs": 10,
        "batch_size": 256,
        "sequence_stride": 12,
        "max_sequences": 120000,
        "num_units": (128, 128),
        "embedding_dim": 96,
        "learning_rate": 0.001,
    },
    "balanced": {
        "epochs": 60,
        "batch_size": 128,
        "sequence_stride": 4,
        "max_sequences": 250000,
        "num_units": (256, 256),
        "embedding_dim": 128,
        "learning_rate": 0.001,
    },
    "quality": {
        "epochs": 150,
        "batch_size": 64,
        "sequence_stride": 1,
        "max_sequences": None,
        "num_units": (256, 256),
        "embedding_dim": 128,
        "learning_rate": 0.0007,
    },
}


def get_training_preset(name="balanced"):
    preset_name = (name or "balanced").lower()
    if preset_name not in TRAINING_PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {', '.join(TRAINING_PRESETS)}"
        )
    return TRAINING_PRESETS[preset_name].copy()


def _to_model_input(model, seed_tokens, vocabulary_size):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 3:
        one_hot = keras.utils.to_categorical(seed_tokens, num_classes=vocabulary_size)
        return one_hot[np.newaxis, ...]

    return np.array(seed_tokens, dtype=np.int32)[np.newaxis, ...]


def _sample_next(probabilities, temperature=0.9):
    epsilon = 1e-8
    adjusted = np.log(np.clip(probabilities, epsilon, 1.0)) / max(temperature, epsilon)
    adjusted = np.exp(adjusted)
    adjusted = adjusted / np.sum(adjusted)
    return int(np.random.choice(np.arange(len(adjusted)), p=adjusted))


def _rollout_tokens(model, seed_tokens, steps, vocabulary_size, temperature=0.9):
    generated = []
    context = list(seed_tokens)
    max_len = len(seed_tokens)

    for _ in range(steps):
        model_input = _to_model_input(model, context[-max_len:], vocabulary_size)
        probabilities = model.predict(model_input, verbose=0)[0]
        next_token = _sample_next(probabilities, temperature=temperature)
        generated.append(next_token)
        context.append(next_token)

    return generated


def _load_index_to_symbol(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as fp:
        mapping = json.load(fp)
    return {int(value): key for key, value in mapping.items()}


def _rhythmic_diversity(symbol_sequences):
    durations = []

    for symbols in symbol_sequences:
        i = 0
        while i < len(symbols):
            symbol = symbols[i]
            if symbol in {"_", "/"}:
                i += 1
                continue

            duration = 1
            j = i + 1
            while j < len(symbols) and symbols[j] == "_":
                duration += 1
                j += 1

            durations.append(duration)
            i = j

    if not durations:
        return 0.0

    unique, counts = np.unique(np.array(durations), return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    normalizer = np.log(len(unique) + 1e-12)
    if normalizer <= 0:
        return 0.0
    return float(entropy / normalizer)


def _music_quality_metrics(model, x_eval, mapping_path, batch_size):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 3:
        vocabulary_size = int(input_shape[-1])
    else:
        vocabulary_size = int(model.output_shape[-1])

    index_to_symbol = _load_index_to_symbol(mapping_path)

    num_rollouts = min(16, len(x_eval))
    rollout_steps = 128
    generated_sequences = []

    for i in range(num_rollouts):
        seed = x_eval[i].tolist()
        generated = _rollout_tokens(
            model=model,
            seed_tokens=seed,
            steps=rollout_steps,
            vocabulary_size=vocabulary_size,
            temperature=0.9,
        )
        generated_sequences.append(generated)

    symbol_sequences = [
        [index_to_symbol.get(token, "/") for token in seq]
        for seq in generated_sequences
    ]

    flat_tokens = [token for seq in generated_sequences for token in seq]
    if len(flat_tokens) > 1:
        repetition_rate = float(
            np.mean(np.array(flat_tokens[1:]) == np.array(flat_tokens[:-1]))
        )
    else:
        repetition_rate = 0.0

    pitch_values = []
    for symbols in symbol_sequences:
        for symbol in symbols:
            if symbol.isdigit():
                pitch_values.append(int(symbol))

    if pitch_values:
        pitch_range = int(max(pitch_values) - min(pitch_values))
        pitch_diversity = float(len(set(pitch_values)) / len(pitch_values))
    else:
        pitch_range = 0
        pitch_diversity = 0.0

    rhythmic_diversity = _rhythmic_diversity(symbol_sequences)

    return {
        "repetition_rate": repetition_rate,
        "pitch_range_semitones": pitch_range,
        "pitch_diversity": pitch_diversity,
        "rhythmic_diversity": rhythmic_diversity,
        "evaluation_rollouts": num_rollouts,
        "rollout_steps": rollout_steps,
    }


def evaluate_model(model, x_eval, y_eval, batch_size=64, mapping_path=MAPPING_PATH):
    """Return token-level and music-quality metrics on held-out validation data."""
    val_loss, val_accuracy = model.evaluate(x_eval, y_eval, batch_size=batch_size, verbose=0)

    probabilities = model.predict(x_eval, batch_size=batch_size, verbose=0)
    top3_indices = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    top3_accuracy = float(np.mean(np.any(top3_indices == y_eval[:, None], axis=1)))
    perplexity = float(np.exp(val_loss))

    quality_metrics = _music_quality_metrics(
        model=model,
        x_eval=x_eval,
        mapping_path=mapping_path,
        batch_size=batch_size,
    )

    metrics = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "top3_accuracy": top3_accuracy,
        "perplexity": perplexity,
    }
    metrics.update(quality_metrics)
    return metrics


def train_model(
    output_units=OUTPUT_UNITS,
    num_units=(256, 256),
    embedding_dim=128,
    loss="sparse_categorical_crossentropy",
    learning_rate=0.001,
    epochs=150,
    batch_size=64,
    sequence_length=SEQUENCE_LENGTH,
    sequence_stride=1,
    max_sequences=None,
    validation_size=0.1,
    model_path=MODEL_PATH,
    preset=None,
):
    if preset:
        preset_values = get_training_preset(preset)
        epochs = preset_values["epochs"]
        batch_size = preset_values["batch_size"]
        sequence_stride = preset_values["sequence_stride"]
        max_sequences = preset_values["max_sequences"]
        num_units = preset_values["num_units"]
        embedding_dim = preset_values["embedding_dim"]
        learning_rate = preset_values["learning_rate"]

    inputs, targets = generate_training_sequences(
        sequence_length=sequence_length,
        stride=sequence_stride,
        max_sequences=max_sequences,
        one_hot_inputs=False,
    )

    vocabulary_size = int(max(output_units, int(targets.max()) + 1))

    x_train, x_val, y_train, y_val = train_test_split(
        inputs,
        targets,
        test_size=validation_size,
        random_state=42,
        shuffle=True,
    )

    model = build_model(
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        num_units=num_units,
        embedding_dim=embedding_dim,
        loss=loss,
        learning_rate=learning_rate,
        dropout_rate=0.3,
        recurrent_dropout=0.1,
        label_smoothing=0.05,
        clipnorm=1.0,
        use_embedding=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.save(str(model_path))

    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    history_dict["final_eval"] = evaluate_model(
        model=model,
        x_eval=x_val,
        y_eval=y_val,
        batch_size=batch_size,
    )

    return model, history_dict


def save_history(history, output_path):
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
