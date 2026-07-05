# Music Generation LSTM Model

This project trains an LSTM on symbolic music data and generates new melodies as MIDI.

## Project Flow

1. Preprocess `.krn` songs into token sequences (notes, rests, sustain markers)
2. Build a symbol mapping and training dataset
3. Train a next-token LSTM model
4. Generate token sequences from a seed
5. Convert generated tokens back to MIDI

## Key Modules

- `src/components/config.py`: Centralized paths and constants
- `src/components/preprocess.py`: Dataset loading, transposition, encoding, mapping, sequence generation
- `src/components/model.py`: LSTM model architecture
- `src/components/train.py`: Training loop with validation and callbacks
- `src/components/generate.py`: Melody generation and MIDI export

Backward compatibility is preserved through `main/PreProcess/preprocess.py`, which now wraps the reusable module implementation.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python scripts/preprocess_data.py
```

Train model:

```bash
python scripts/train_model.py
```

Generate melody:

```bash
python scripts/generate_melody.py
```

## Important Improvements Included

- Cross-platform paths (no hardcoded Windows absolute paths)
- Dense sequence sampling for training (`stride=1` by default)
- Validation split and training callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`)
- Numerical stability in temperature sampling
- Fixed logger directory creation behavior

## Producing Better Songs

Use these settings as a starting point for higher-quality melodies:

- Train longer with validation tracking: `epochs=120` to `200`
- Keep dense sampling: `sequence_stride=1`
- Use stacked recurrent layers: default is now `(256, 256)` with embedding input
- Use controlled decoding:
  - `temperature`: `0.75` to `0.95`
  - `top_k`: `15` to `35`
  - `top_p`: `0.88` to `0.95`
  - `repetition_penalty`: `1.08` to `1.2`

If output is too repetitive, increase `repetition_penalty` and reduce `top_p` slightly.
If output is too random, reduce `temperature` and `top_k`.

## Output Files

- `main/mapping.json`: Symbol-to-index mapping
- `main/model.h5`: Trained model
- `main/mel.mid`: Generated MIDI
- `main/training_history.json`: Saved training history
