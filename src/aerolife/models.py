"""Compact recurrent regressors (Keras)."""

from __future__ import annotations

import os
import time

import numpy as np


def set_seeds(seed: int, deterministic: bool = True) -> None:
    import tensorflow as tf
    import keras

    keras.utils.set_random_seed(seed)  # python, numpy, tf
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # older TF or already set
            pass
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def build_rnn(seq_len: int, n_features: int, cell: str = "lstm", units: int = 32,
              dropout: float = 0.2, dense_units: int = 16, loss: str = "huber",
              learning_rate: float = 1e-3):
    """Input(L, F) -> {LSTM|GRU|SimpleRNN}(units) -> Dropout -> Dense(ReLU) -> Dense(1, softplus).

    Softplus keeps the RUL output non-negative without the dead-unit risk of ReLU.
    """
    import keras
    from keras import layers

    rnn = {"lstm": layers.LSTM, "gru": layers.GRU, "simplernn": layers.SimpleRNN}[cell]
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        rnn(units),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1, activation="softplus"),
    ], name=f"aerolife_{cell}")
    losses = {"mae": "mae", "mse": "mse", "huber": keras.losses.Huber(delta=1.0)}
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=losses[loss], metrics=["mae"])
    return model


def train(model, X_tr, y_tr, X_val, y_val, *, epochs: int, batch_size: int, patience: int,
          checkpoint_path: str):
    """Early stopping, LR reduction and best-checkpoint saving, all on validation units."""
    import keras

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 3)),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]
    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=batch_size, shuffle=True, callbacks=cbs, verbose=2)
    return hist.history, time.perf_counter() - t0


def predict(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(0)
    return model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
