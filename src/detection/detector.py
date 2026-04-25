"""
src/detection/detector.py
=========================
Phát hiện Change-Point trên chuỗi thời gian dài (Sliding Window).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ChangePointDetector:
    def __init__(self, model: tf.keras.Model, classes: np.ndarray, config):
        self.model = model
        self.classes = classes
        self.config = config
        
        # Determine transition class indices (classes containing '->')
        trans_sym = "->"
        self.transition_indices = [i for i, c in enumerate(self.classes) if trans_sym in c]
        self.null_indices = [i for i, c in enumerate(self.classes) if trans_sym not in c]

    def _preprocess_sequence(self, seq_3d: np.ndarray):
        sq = np.square(seq_3d)
        combined = np.concatenate([seq_3d, sq], axis=2)
        datamin = np.min(combined, axis=(1, 2), keepdims=True)
        datamax = np.max(combined, axis=(1, 2), keepdims=True)
        denom = datamax - datamin
        denom[denom == 0] = 1e-8
        normalized = 2 * (combined - datamin) / denom - 1
        return normalized  # Mạng Conv1D nhận (N, W, C)

    def detect(self, seq_2d: np.ndarray):
        """
        Input: sequence 2D numpy array (N_total_samples, 3)
        Output:
            cp_estimated_idx: list of sample indices where CPs are detected
            p_smoothed: probability scores
            window_centers: center indices of sliding windows
        """
        n_total = len(seq_2d)
        wl = self.config.WINDOW_LENGTH
        step = self.config.SLIDE_STEP
        
        if n_total < wl:
            raise ValueError(f"Sequence ({n_total}) quá ngắn so với window ({wl})")

        n_windows = (n_total - wl) // step + 1
        windows = np.zeros((n_windows, wl, 3), dtype=np.float32)
        window_centers = np.zeros(n_windows, dtype=int)

        for i in range(n_windows):
            start = i * step
            windows[i] = seq_2d[start:start + wl]
            window_centers[i] = start + wl // 2

        x_windows = self._preprocess_sequence(windows)
        
        # Predict using batching to avoid OOM
        logits = self.model.predict(x_windows, batch_size=self.config.DETECT_BATCH_SIZE, verbose=1)
        probs = tf.nn.softmax(logits).numpy()

        if len(self.transition_indices) > 0:
            p_transition = probs[:, self.transition_indices].sum(axis=1)
        else:
            p_transition = 1 - probs[:, self.null_indices].sum(axis=1)

        # Smoothing (Algorithm 1)
        # B1: Lấy nhãn cứng (L_i = 1 nếu prob > 0.5)
        L = (p_transition >= 0.5).astype(float)
        
        # B2: Tính Trung bình trượt (Moving Average)
        # Giống paper: Smooth qua cửa sổ xung quanh n_w
        window_conv = max(1, wl // step)
        kernel = np.ones(window_conv) / window_conv
        p_smoothed = np.convolve(L, kernel, mode="same")

        # B3: Tìm các phân đoạn có L_bar >= \gamma (với \gamma = 0.5)
        gamma = 0.5
        activated = (p_smoothed >= gamma)
        
        # Tìm ranh giới các phân đoạn (segments)
        diff = np.diff(np.concatenate(([0], activated.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        peaks = []
        for s, e in zip(starts, ends):
            # B4: Argmax của L_bar trong từng phân đoạn
            segment_vals = p_smoothed[s:e]
            if len(segment_vals) > 0:
                local_peak = s + np.argmax(segment_vals)
                peaks.append(local_peak)
                
        peaks = np.array(peaks)

        cp_estimated_idx = window_centers[peaks] if len(peaks) > 0 else np.array([])
        return cp_estimated_idx, p_smoothed, window_centers, probs, peaks

    def plot_detection(self, seq: np.ndarray, times: np.ndarray, labels_df, 
                       cp_estimated_idx, true_cps_idx, p_transition, p_smoothed, 
                       window_centers, probs, peaks, save_path, title_info):
        
        n_total = len(seq)
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.4)

        # 1. Raw Time Series
        ax1 = fig.add_subplot(gs[0])
        t_axis = np.arange(n_total)
        ax1.plot(t_axis, seq[:, 0], color="#2196F3", lw=0.7, alpha=0.9, label="X-axis")
        ax1.plot(t_axis, seq[:, 1], color="#4CAF50", lw=0.7, alpha=0.9, label="Y-axis")
        ax1.plot(t_axis, seq[:, 2], color="#FF9800", lw=0.7, alpha=0.9, label="Z-axis")

        for cp in true_cps_idx:
            ax1.axvline(x=cp, color="#F44336", lw=2.0, linestyle="--", alpha=0.9, label="True CP" if cp == true_cps_idx[0] else "")
        for cp in cp_estimated_idx:
            ax1.axvline(x=cp, color="#00BCD4", lw=1.5, linestyle="-.", alpha=0.9, label="Detected CP" if cp == cp_estimated_idx[0] else "")

        ymin, ymax = ax1.get_ylim()
        for i, row in labels_df.iterrows():
            s_idx = np.searchsorted(times, row["start"])
            e_idx = np.searchsorted(times, row["end"])
            s_idx = np.clip(s_idx, 0, n_total - 1)
            e_idx = np.clip(e_idx, 0, n_total - 1)
            mid = (s_idx + e_idx) // 2
            ax1.text(mid, ymax * 0.85, row["state"], ha="center", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7, edgecolor="gray"))

        ax1.set_title(f"Raw Accelerometer Signal — {title_info}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Acceleration (g)")
        ax1.legend(loc="upper right", fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.2)

        # 2. Probability Score
        ax2 = fig.add_subplot(gs[1])
        x_win = window_centers[:len(p_transition)]
        ax2.fill_between(x_win, p_transition, alpha=0.3, color="#9C27B0", label="P(transition) raw")
        ax2.plot(x_win, p_smoothed, color="#9C27B0", lw=1.5, label="Smoothed")
        ax2.axhline(y=self.config.PEAK_HEIGHT, color="gray", lw=1.0, linestyle=":", alpha=0.7, label=f"Threshold={self.config.PEAK_HEIGHT}")

        if len(peaks) > 0:
            ax2.scatter(window_centers[peaks], p_smoothed[peaks], color="#FF5722", zorder=5,
                        s=60, label=f"Detected peaks ({len(peaks)})", marker="^")
        for cp in true_cps_idx:
            ax2.axvline(x=cp, color="#F44336", lw=1.5, linestyle="--", alpha=0.7)

        ax2.set_title("Transition Probability Score (Sliding Window)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("P(transition)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.2)

        # 3. Label Distribution
        ax3 = fig.add_subplot(gs[2])
        predicted_labels_idx = np.argmax(probs, axis=1)
        unique_labels, label_counts = np.unique(predicted_labels_idx, return_counts=True)
        bar_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        bars = ax3.bar([self.classes[u] for u in unique_labels], label_counts, color=bar_colors, edgecolor="white", linewidth=0.8)
        
        ax3.set_title("Phân bố nhãn dự đoán trên tất cả windows", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Predicted Activity/Transition")
        ax3.set_ylabel("Số lượng windows")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        for bar, count in zip(bars, label_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontsize=8)
        ax3.grid(True, alpha=0.2, axis="y")

        plt.suptitle(
            f"AutoCPD — Change-Point Detection\n{title_info}\n"
            f"Detected: {len(cp_estimated_idx)} | True: {len(true_cps_idx)}",
            fontsize=12, fontweight="bold", y=1.01
        )

        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()

    def get_ground_truth_cps(self, labels_df, times, n_total):
        """Chuyển timestamps thành index."""
        true_cps_idx = []
        for i in range(len(labels_df) - 1):
            t_cp = labels_df.iloc[i]["end"]
            idx_cp = np.searchsorted(times, t_cp)
            idx_cp = np.clip(idx_cp, 0, n_total - 1)
            true_cps_idx.append(idx_cp)
        return true_cps_idx
