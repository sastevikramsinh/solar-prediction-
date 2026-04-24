"""Artificial neural network model for solar power prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


class ANNModel:
    """Dense ANN regressor."""

    def __init__(self) -> None:
        """Initialize ANN wrapper."""
        self.model = None

    def build_model(self, input_dim: int) -> Sequential:
        """Build ANN architecture.

        Args:
            input_dim: Number of input features.

        Returns:
            Compiled Keras model.
        """
        model = Sequential(
            [
                Dense(256, activation="relu", input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        self.model = model
        return model

    def train(
        self,
        x_train_sc: np.ndarray,
        y_train_sc: np.ndarray,
        x_val_sc: np.ndarray,
        y_val_sc: np.ndarray,
    ) -> tuple[Sequential, object]:
        """Train ANN with early stopping and LR scheduling.

        Args:
            x_train_sc: Scaled train features.
            y_train_sc: Scaled train target.
            x_val_sc: Scaled validation features.
            y_val_sc: Scaled validation target.

        Returns:
            Trained model and history object.
        """
        if self.model is None:
            self.build_model(input_dim=x_train_sc.shape[1])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
            ModelCheckpoint("./models/ann_best.keras", save_best_only=True),
        ]

        history = self.model.fit(
            x_train_sc,
            y_train_sc,
            validation_data=(x_val_sc, y_val_sc),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        output_path = Path("./outputs/ann_training_history.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("ANN Training History")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {output_path.as_posix()}")

        # Ensure best checkpoint is loaded before returning.
        self.model = load_model("./models/ann_best.keras")
        return self.model, history

    def predict(self, x_test_sc: np.ndarray, y_scaler: MinMaxScaler) -> np.ndarray:
        """Predict and inverse-transform output back to kW scale.

        Args:
            x_test_sc: Scaled test features.
            y_scaler: Target scaler fit on train target.

        Returns:
            Predictions in original AC power scale.
        """
        preds_scaled = self.model.predict(x_test_sc, verbose=0)
        return y_scaler.inverse_transform(preds_scaled).ravel()

    def save(self, path: str = "./models/ann_final.keras") -> None:
        """Save final ANN model.

        Args:
            path: Model output path.

        Returns:
            None.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)
        print(f"[SAVED] {out_path.as_posix()}")

