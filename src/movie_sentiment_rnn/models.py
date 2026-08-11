"""Configurable TensorFlow recurrent architectures and training callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Architecture = Literal["simple_rnn", "lstm", "gru"]
Task = Literal["classification", "regression"]


class HyperParameters(Protocol):
    """Subset of the Keras Tuner hyperparameter interface used by builders."""

    def Choice(self, name: str, values: list[int] | list[float]) -> int | float:
        """Choose one discrete hyperparameter value."""
        ...

    def Float(self, name: str, min_value: float, max_value: float, step: float) -> float:
        """Choose one stepped floating-point hyperparameter value."""
        ...


@dataclass(frozen=True)
class ModelConfig:
    """Architecture, task, vocabulary, sequence, and output configuration."""

    architecture: Architecture = "gru"
    task: Task = "classification"
    input_length: int = 100
    vocab_size: int = 10000
    l2_value: float = 0.001


class DefaultHyperParameters:
    """Small Keras-Tuner compatible object for deterministic default builds."""

    def Choice(self, name: str, values: list[int] | list[float]) -> int | float:
        """Return the first discrete value for a deterministic default build."""
        del name
        return values[0]

    def Float(self, name: str, min_value: float, max_value: float, step: float) -> float:
        """Return the lower bound for a deterministic default build."""
        del name, max_value, step
        return min_value


def _keras_imports():
    try:
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            LearningRateScheduler,
            ReduceLROnPlateau,
        )
        from tensorflow.keras.layers import (
            GRU,
            LSTM,
            BatchNormalization,
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            Input,
            LeakyReLU,
            SimpleRNN,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model construction. Install notebook dependencies with "
            "`python -m pip install -r requirements-notebook.txt`."
        ) from exc

    return {
        "Adam": Adam,
        "BatchNormalization": BatchNormalization,
        "Bidirectional": Bidirectional,
        "Dense": Dense,
        "Dropout": Dropout,
        "EarlyStopping": EarlyStopping,
        "Embedding": Embedding,
        "GRU": GRU,
        "Input": Input,
        "LSTM": LSTM,
        "LearningRateScheduler": LearningRateScheduler,
        "LeakyReLU": LeakyReLU,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "Sequential": Sequential,
        "SimpleRNN": SimpleRNN,
        "l2": l2,
    }


def learning_rate_schedule(epoch: int, learning_rate: float) -> float:
    """Halve the learning rate after each completed ten-epoch interval."""
    if epoch and epoch % 10 == 0:
        return learning_rate * 0.5
    return learning_rate


def build_callbacks(checkpoint_path: str | None = None):
    """Build early-stopping, scheduling, and optional checkpoint callbacks."""
    keras = _keras_imports()
    callbacks = [
        keras["EarlyStopping"](monitor="val_loss", patience=10, restore_best_weights=True),
        keras["ReduceLROnPlateau"](monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras["LearningRateScheduler"](learning_rate_schedule),
    ]
    if checkpoint_path:
        from tensorflow.keras.callbacks import ModelCheckpoint

        callbacks.append(ModelCheckpoint(checkpoint_path, save_best_only=True))
    return callbacks


def build_model(config: ModelConfig, hp: HyperParameters | None = None):
    """Build and compile the configured recurrent sentiment model."""
    hp = hp or DefaultHyperParameters()
    keras = _keras_imports()
    recurrent_layer = {
        "simple_rnn": keras["SimpleRNN"],
        "lstm": keras["LSTM"],
        "gru": keras["GRU"],
    }[config.architecture]

    model = keras["Sequential"]()
    model.add(keras["Input"](shape=(config.input_length,)))
    model.add(
        keras["Embedding"](
            input_dim=config.vocab_size,
            output_dim=hp.Choice("embed_dim", [128, 256, 300]),
            embeddings_initializer="uniform",
        )
    )

    if config.task == "classification":
        _add_classification_layers(model, recurrent_layer, keras, hp, config)
        model.add(keras["Dense"](1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
        learning_rates = [5e-4, 1e-4, 5e-5]
    else:
        _add_regression_layers(model, recurrent_layer, keras, hp, config)
        model.add(keras["Dense"](1, activation="linear"))
        loss = "mean_squared_error"
        metrics = ["mae"]
        learning_rates = [1e-3, 1e-4, 5e-5]

    model.compile(
        optimizer=keras["Adam"](learning_rate=hp.Choice("lr", learning_rates)),
        loss=loss,
        metrics=metrics,
    )
    return model


def _add_classification_layers(
    model,
    recurrent_layer,
    keras: dict[str, object],
    hp,
    config,
) -> None:
    regularizer = keras["l2"](config.l2_value)
    model.add(
        keras["Bidirectional"](
            recurrent_layer(
                units=hp.Choice("rnn_units_1", [128, 256]),
                return_sequences=True,
                activation="tanh",
                kernel_regularizer=regularizer,
            )
        )
    )
    model.add(keras["BatchNormalization"]())
    model.add(keras["Dropout"](hp.Float("dropout_rnn_1", 0.3, 0.5, step=0.1)))
    model.add(
        keras["Bidirectional"](
            recurrent_layer(
                units=hp.Choice("rnn_units_2", [64, 128]),
                return_sequences=False,
                activation="tanh",
                kernel_regularizer=regularizer,
            )
        )
    )
    model.add(keras["BatchNormalization"]())
    model.add(keras["Dropout"](hp.Float("dropout_rnn_2", 0.3, 0.5, step=0.1)))
    _add_dense_block(model, keras, hp, "dense_units_1", "dropout_dense_1", [128, 256])
    _add_dense_block(model, keras, hp, "dense_units_2", "dropout_dense_2", [64, 128])


def _add_regression_layers(model, recurrent_layer, keras: dict[str, object], hp, config) -> None:
    regularizer = keras["l2"](config.l2_value)
    model.add(
        recurrent_layer(
            units=hp.Choice("rnn_units_1", [128, 256]),
            return_sequences=True,
            activation="tanh",
            kernel_regularizer=regularizer,
        )
    )
    model.add(keras["BatchNormalization"]())
    model.add(keras["Dropout"](hp.Float("dropout_rnn_1", 0.3, 0.5, step=0.1)))
    model.add(
        recurrent_layer(
            units=hp.Choice("rnn_units_2", [64, 128]),
            return_sequences=False,
            activation="tanh",
            kernel_regularizer=regularizer,
        )
    )
    model.add(keras["BatchNormalization"]())
    model.add(keras["Dropout"](hp.Float("dropout_rnn_2", 0.2, 0.5, step=0.1)))
    _add_dense_block(model, keras, hp, "dense_units_1", "dropout_dense_1", [64, 128, 256])
    _add_dense_block(model, keras, hp, "dense_units_2", "dropout_dense_2", [32, 64])


def _add_dense_block(
    model,
    keras: dict[str, object],
    hp,
    units_name: str,
    dropout_name: str,
    unit_choices: list[int],
) -> None:
    model.add(
        keras["Dense"](
            units=hp.Choice(units_name, unit_choices),
            kernel_regularizer=keras["l2"](0.001),
        )
    )
    model.add(keras["LeakyReLU"](negative_slope=0.1))
    model.add(keras["BatchNormalization"]())
    model.add(keras["Dropout"](hp.Float(dropout_name, 0.2, 0.5, step=0.1)))


def build_classifier(architecture: Architecture = "gru", **kwargs):
    """Build a binary sentiment classifier for the requested architecture."""
    return build_model(ModelConfig(architecture=architecture, task="classification", **kwargs))


def build_regressor(architecture: Architecture = "gru", **kwargs):
    """Build a sentiment-score regressor for the requested architecture."""
    return build_model(ModelConfig(architecture=architecture, task="regression", **kwargs))
