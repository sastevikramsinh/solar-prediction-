"""CNN-LSTM hybrid model for time-series solar forecasting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

WINDOW_SIZE = 12


def create_sequences(
    x: np.ndarray, y: np.ndarray, window_size: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-window temporal sequences for recurrent training.

    Args:
        x: Feature matrix.
        y: Target array.
        window_size: Number of historical timesteps.

    Returns:
        Sequence tensor and aligned targets.
    """
    sequences, targets = [], []
    for i in range(window_size, len(x)):
        sequences.append(x[i - window_size : i])
        targets.append(y[i])
    return np.array(sequences), np.array(targets)


class CNNLSTMModel:
    """CNN-LSTM regressor for 15-minute PV generation sequence learning."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        """Initialize model wrapper.

        Args:
            window_size: Input sequence length.

        Returns:
            None.
        """
        self.window_size = window_size
        self.model = None

    def build_model(self, n_features: int, window_size: int = WINDOW_SIZE) -> Model:
        """Build CNN-LSTM network via functional API.

        Args:
            n_features: Number of input features.
            window_size: Sequence length.

        Returns:
            Compiled Keras model.
        """
        inputs = Input(shape=(window_size, n_features))
        x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
        x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
        model.summary()
        self.model = model
        return model

    def train(
        self,
        x_train_sc: np.ndarray,
        y_train_sc: np.ndarray,
        x_val_sc: np.ndarray,
        y_val_sc: np.ndarray,
        x_test_sc: np.ndarray,
        y_test_sc: np.ndarray,
    ) -> tuple[Model, object, np.ndarray]:
        """Train CNN-LSTM on sequenced train/validation sets.

        Args:
            x_train_sc: Scaled train features.
            y_train_sc: Scaled train target.
            x_val_sc: Scaled validation features.
            y_val_sc: Scaled validation target.
            x_test_sc: Scaled test features.
            y_test_sc: Scaled test target.

        Returns:
            Trained model, fit history, and sequenced test features.
        """
        x_tr_seq, y_tr_seq = create_sequences(x_train_sc, y_train_sc, self.window_size)
        x_v_seq, y_v_seq = create_sequences(x_val_sc, y_val_sc, self.window_size)
        x_test_seq, _ = create_sequences(x_test_sc, y_test_sc, self.window_size)

        if self.model is None:
            self.build_model(n_features=x_train_sc.shape[1], window_size=self.window_size)

        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
            ),
            ModelCheckpoint("./models/cnn_lstm_best.keras", save_best_only=True),
        ]

        history = self.model.fit(
            x_tr_seq,
            y_tr_seq,
            validation_data=(x_v_seq, y_v_seq),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        out_path = Path("./outputs/cnn_lstm_training_history.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("CNN-LSTM Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {out_path.as_posix()}")

        self.model = load_model("./models/cnn_lstm_best.keras")
        return self.model, history, x_test_seq

    def predict(self, x_test_seq: np.ndarray, y_scaler: MinMaxScaler) -> np.ndarray:
        """Predict and inverse-scale output values.

        Args:
            x_test_seq: Sequenced scaled test features.
            y_scaler: Target scaler.

        Returns:
            Predictions in original kW scale.
        """
        preds_scaled = self.model.predict(x_test_seq, verbose=0).ravel()
        return y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    def save(self, path: str = "./models/cnn_lstm_final.keras") -> None:
        """Save trained CNN-LSTM model.

        Args:
            path: Output model path.

        Returns:
            None.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)
        print(f"[SAVED] {out_path.as_posix()}")

