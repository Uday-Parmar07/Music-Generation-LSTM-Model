import json

import music21 as m21
import numpy as np
from keras import models
from keras.utils import to_categorical

from src.components.config import MAPPING_PATH, MODEL_PATH, OUTPUT_MIDI_PATH, SEQUENCE_LENGTH


class MelodyGenerator:
    """Wrapper for loading a trained model and generating melodies."""

    def __init__(self, model_path=MODEL_PATH, mapping_path=MAPPING_PATH):
        self.model_path = str(model_path)
        self.mapping_path = str(mapping_path)

        self.model = models.load_model(self.model_path)
        with open(self.mapping_path, "r", encoding="utf-8") as fp:
            self._mappings = json.load(fp)

        self._index_to_symbol = {value: key for key, value in self._mappings.items()}
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(
        self,
        seed,
        num_steps,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        seed_tokens = seed.split()
        melody = seed_tokens.copy()
        seed_tokens = self._start_symbols + seed_tokens

        seed_as_int = [self._mappings[symbol] for symbol in seed_tokens]

        for _ in range(num_steps):
            seed_as_int = seed_as_int[-max_sequence_length:]

            model_input = self._prepare_model_input(seed_as_int)
            probabilities = self.model.predict(model_input, verbose=0)[0]
            output_int = self._sample_with_controls(
                probabilities=probabilities,
                recent_tokens=seed_as_int,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            seed_as_int.append(output_int)
            output_symbol = self._index_to_symbol[output_int]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody

    def _prepare_model_input(self, seed_as_int):
        """Support both legacy one-hot models and newer embedding-based models."""
        input_shape = self.model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) == 3:
            onehot_seed = to_categorical(seed_as_int, num_classes=len(self._mappings))
            return onehot_seed[np.newaxis, ...]

        return np.array(seed_as_int, dtype=np.int32)[np.newaxis, ...]

    @staticmethod
    def _sample_with_controls(
        probabilities,
        recent_tokens,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        """Sample from a distribution using temperature, top-k/top-p, and repetition control."""
        epsilon = 1e-8
        adjusted = np.array(probabilities, dtype=np.float64)

        if repetition_penalty > 1.0:
            for token in recent_tokens[-32:]:
                adjusted[token] /= repetition_penalty

        adjusted = np.log(np.clip(adjusted, epsilon, 1.0)) / max(temperature, epsilon)
        adjusted = np.exp(adjusted)
        adjusted = adjusted / np.sum(adjusted)

        if top_k is not None and top_k > 0 and top_k < len(adjusted):
            top_indices = np.argpartition(adjusted, -top_k)[-top_k:]
            filtered = np.zeros_like(adjusted)
            filtered[top_indices] = adjusted[top_indices]
            adjusted = filtered / np.sum(filtered)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(adjusted)[::-1]
            sorted_probs = adjusted[sorted_indices]
            cumulative = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumulative, top_p) + 1
            nucleus_indices = sorted_indices[:cutoff]
            filtered = np.zeros_like(adjusted)
            filtered[nucleus_indices] = adjusted[nucleus_indices]
            adjusted = filtered / np.sum(filtered)

        choices = np.arange(len(adjusted))
        return int(np.random.choice(choices, p=adjusted))

    @staticmethod
    def save_melody(melody, step_duration=0.25, output_path=OUTPUT_MIDI_PATH, output_format="midi"):
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    if start_symbol == "r":
                        event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(event)
                    step_counter = 1

                start_symbol = symbol
            else:
                step_counter += 1

        stream.write(output_format, str(output_path))
