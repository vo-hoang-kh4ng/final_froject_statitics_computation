"""
src/training/trainer.py
=======================
Xử lý logic compile, fit model, logging và saving.
"""

import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model: tf.keras.Model,
                 learning_rate: float,
                 decay_steps: int,
                 decay_rate: float,
                 epochs: int,
                 batch_size: int,
                 validation_split: float,
                 early_stop_patience: int):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Optimizer schedule
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
        self.early_stop_patience = early_stop_patience

    def compile(self):
        """Compile mô hình."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr_schedule),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray, log_csv_path: pathlib.Path, best_model_path: pathlib.Path):
        """Huấn luyện mô hình với callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.early_stop_patience,
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(log_csv_path)),
            tf.keras.callbacks.ModelCheckpoint(
                str(best_model_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        print(f"[TRAIN] Bắt đầu huấn luyện ({self.epochs} epochs).")
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    @staticmethod
    def plot_history(history_dict: dict, save_path: pathlib.Path):
        """Vẽ learning curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("AutoCPD - Training Results", fontsize=14, fontweight="bold")

        epochs_range = range(1, len(history_dict["accuracy"]) + 1)

        ax1.plot(epochs_range, history_dict["accuracy"], label="Training Accuracy", color="#2196F3")
        ax1.plot(epochs_range, history_dict["val_accuracy"], label="Validation Accuracy", color="#FF5722")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history_dict["loss"], label="Training Loss", color="#2196F3")
        ax2.plot(epochs_range, history_dict["val_loss"], label="Validation Loss", color="#FF5722")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
