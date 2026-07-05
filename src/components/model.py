import keras


def build_model(
    vocabulary_size,
    sequence_length=None,
    num_units=(256, 256),
    embedding_dim=128,
    loss="sparse_categorical_crossentropy",
    learning_rate=0.001,
    dropout_rate=0.3,
    recurrent_dropout=0.1,
    label_smoothing=0.0,
    clipnorm=1.0,
    use_embedding=True,
):
    """Build and compile an LSTM model for next-token prediction."""
    if not num_units:
        raise ValueError("num_units must contain at least one layer size")

    if use_embedding:
        inputs = keras.layers.Input(shape=(sequence_length,), dtype="int32")
        x = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            mask_zero=False,
        )(inputs)
    else:
        inputs = keras.layers.Input(shape=(None, vocabulary_size))
        x = inputs

    for layer_index, units in enumerate(num_units):
        is_last = layer_index == len(num_units) - 1
        x = keras.layers.LSTM(
            units,
            return_sequences=not is_last,
            recurrent_dropout=recurrent_dropout,
        )(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(vocabulary_size, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    if loss == "sparse_categorical_crossentropy":
        # Some Keras versions do not support label_smoothing for sparse CE.
        if label_smoothing > 0:
            try:
                loss_fn = keras.losses.SparseCategoricalCrossentropy(
                    label_smoothing=label_smoothing
                )
            except TypeError:
                loss_fn = keras.losses.SparseCategoricalCrossentropy()
        else:
            loss_fn = keras.losses.SparseCategoricalCrossentropy()
    else:
        loss_fn = loss

    model.compile(
        loss=loss_fn,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        metrics=["accuracy"],
    )

    return model
