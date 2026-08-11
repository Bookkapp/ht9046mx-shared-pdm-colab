from __future__ import annotations

import importlib.util

from .config import PipelineConfig


def require_tensorflow():
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError(
            "TensorFlow is required to train or infer the LSTM Autoencoder. "
            "Install project requirements in a virtual environment: python -m pip install -r requirements.txt"
        )
    import tensorflow as tf
    return tf


def build_lstm_autoencoder(config: PipelineConfig, feature_count: int):
    tf = require_tensorflow()
    inputs = tf.keras.Input(shape=(config.window_rows, feature_count), name="compressor_window")
    x = tf.keras.layers.LSTM(config.encoder_units[0], return_sequences=True, name="encoder_lstm_1")(inputs)
    latent = tf.keras.layers.LSTM(config.encoder_units[1], name="encoder_latent")(x)
    x = tf.keras.layers.RepeatVector(config.window_rows, name="repeat_latent")(latent)
    x = tf.keras.layers.LSTM(config.decoder_units[0], return_sequences=True, name="decoder_lstm_1")(x)
    x = tf.keras.layers.LSTM(config.decoder_units[1], return_sequences=True, name="decoder_lstm_2")(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(feature_count), name="reconstruction")(x)
    model = tf.keras.Model(inputs, outputs, name="ht9046mx_lstm_autoencoder")
    model.compile(optimizer="adam", loss=config.loss)
    return model
