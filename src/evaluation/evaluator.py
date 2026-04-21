"""
src/evaluation/evaluator.py
===========================
Đánh giá mô hình đã huấn luyện (accuracy, classification report, confusion matrix).
"""

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model_path: pathlib.Path, classes: np.ndarray):
        if not model_path.exists():
            raise FileNotFoundError(f"Model không tồn tại: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        self.classes = classes

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Predict and return metrics and predictions."""
        logits = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(logits, axis=1)
        probs = tf.nn.softmax(logits).numpy()

        overall_acc = np.mean(y_pred == y_test)
        report = classification_report(y_test, y_pred, target_names=self.classes, zero_division=0)
        
        return {
            "accuracy": overall_acc,
            "report": report,
            "y_pred": y_pred,
            "probs": probs
        }

    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, title: str, save_path: pathlib.Path):
        """Vẽ và lưu Confusion Matrix."""
        cm = confusion_matrix(y_test, y_pred)
        n_cls = len(self.classes)

        cm_normalized = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm_normalized / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, data, subtitle, fmt in zip(
            axes, [cm, cm_normalized], ["Số lượng (Count)", "Chuẩn hóa (Normalized)"], ["d", ".2f"]
        ):
            im = ax.imshow(data, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(subtitle, fontsize=12)
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)
            tick_marks = np.arange(n_cls)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)

            short_labels = [c if len(c) <= 12 else c[:10] + ".." for c in self.classes]
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short_labels, fontsize=7)

            thresh = data.max() / 2.0
            for i in range(n_cls):
                for j in range(n_cls):
                    val = data[i, j]
                    txt = format(val, fmt)
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=6)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()

    def analyze_binary_detection(self, y_test: np.ndarray, y_pred: np.ndarray, trans_sym: str = "->"):
        """Đánh giá theo bài toán Binary: Có Change-Point hay KHÔNG có Change-Point."""
        # ground truth có chứa mũi tên -> không?
        is_transition_true = np.array([trans_sym in self.classes[t] for t in y_test])
        # predict có chứa mũi tên -> không?
        is_transition_pred = np.array([trans_sym in self.classes[p] for p in y_pred])

        tp = np.sum(is_transition_true & is_transition_pred)
        tn = np.sum(~is_transition_true & ~is_transition_pred)
        fp = np.sum(~is_transition_true & is_transition_pred)
        fn = np.sum(is_transition_true & ~is_transition_pred)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        binary_acc = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
        
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, 
            "f1": f1, "accuracy": binary_acc
        }
