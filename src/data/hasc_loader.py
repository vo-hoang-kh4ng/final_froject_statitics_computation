"""
src/data/hasc_loader.py
=======================
Chịu trách nhiệm Load, Extract, và Preprocess data HASC.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class HascDataLoader:
    def __init__(self, data_root: pathlib.Path, length: int, size: int, size0: int, num_trim: int):
        self.data_root = data_root
        self.length = length
        self.size = size
        self.size0 = size0
        self.num_trim = num_trim

    def read_hasc_csv(self, csv_path: pathlib.Path) -> pd.DataFrame:
        """Đọc file CSV của HASC (không có header)."""
        return pd.read_csv(
            csv_path,
            comment="#",
            delimiter=",",
            names=["time", "x", "y", "z"]
        )

    def read_hasc_label(self, label_path: pathlib.Path) -> pd.DataFrame:
        """Đọc file .label của HASC."""
        return pd.read_csv(
            label_path,
            comment="#",
            delimiter=",",
            names=["start", "end", "state"]
        )

    def extract_transition_segments(self, csv_path: pathlib.Path, label_path: pathlib.Path):
        """Trích xuất các đoạn time series CÓ change-point."""
        data = self.read_hasc_csv(csv_path)
        label = self.read_hasc_label(label_path)
        n_states = len(label)

        ts_list, cp_list, label_list = [], [], []

        for i in range(n_states - 1):
            s0, e0, state0 = label.iloc[i]["start"], label.iloc[i]["end"], label.iloc[i]["state"]
            s1, e1, state1 = label.iloc[i+1]["start"], label.iloc[i+1]["end"], label.iloc[i+1]["state"]
            transition_label = f"{state0}->{state1}"

            mask0 = (data["time"] >= s0) & (data["time"] <= e0)
            mask1 = (data["time"] >= s1) & (data["time"] <= e1)
            n0 = mask0.sum()
            n1 = mask1.sum()

            if n0 < self.num_trim + self.size or n1 < self.num_trim + self.size:
                continue
            if n0 + n1 < self.length:
                continue

            seg0 = data[mask0][["x", "y", "z"]].to_numpy()
            seg1 = data[mask1][["x", "y", "z"]].to_numpy()
            seg_concat = np.concatenate([seg0, seg1], axis=0)
            total_len = len(seg_concat)

            true_cp = n0
            half = self.length // 2
            n_extracted = 0
            attempts = 0

            while n_extracted < self.size and attempts < 200:
                attempts += 1
                start_min = max(0, true_cp - self.length + self.num_trim)
                start_max = min(total_len - self.length, true_cp - self.num_trim)
                if start_min >= start_max:
                    break
                
                # NumPy random
                start = np.random.randint(start_min, start_max)
                end = start + self.length
                
                if end > total_len:
                    continue
                
                segment = seg_concat[start:end]
                cp_in_seg = true_cp - start
                
                if cp_in_seg < self.num_trim or cp_in_seg > self.length - self.num_trim:
                    continue
                
                ts_list.append(segment)
                cp_list.append(cp_in_seg)
                label_list.append(transition_label)
                n_extracted += 1

        return ts_list, cp_list, label_list

    def extract_null_segments(self, csv_path: pathlib.Path, label_path: pathlib.Path):
        """Trích xuất các đoạn time series KHÔNG có change-point."""
        data = self.read_hasc_csv(csv_path)
        label = self.read_hasc_label(label_path)
        n_states = len(label)

        ts_list, label_list = [], []

        for i in range(n_states):
            s, e, state = label.iloc[i]["start"], label.iloc[i]["end"], label.iloc[i]["state"]
            mask = (data["time"] >= s) & (data["time"] <= e)
            seg = data[mask][["x", "y", "z"]].to_numpy()
            n_seg = len(seg)
            if n_seg < self.length + self.size0:
                continue
            
            starts = np.sort(np.random.choice(range(0, n_seg - self.length), size=self.size0, replace=False))
            for s_idx in starts:
                ts_list.append(seg[s_idx:s_idx + self.length])
                label_list.append(state)

        return ts_list, label_list

    def load_dataset(self, subjects: list, known_classes=None):
        """Load data cho danh sách subjects."""
        all_ts, all_labels = [], []

        for subject in subjects:
            subject_dir = self.data_root / subject
            if not subject_dir.exists():
                print(f"[WARN] Không tìm thấy: {subject_dir}")
                continue

            csv_files = sorted([f for f in os.listdir(subject_dir) if f.startswith("HASC") and f.endswith(".csv")])

            for csv_fname in csv_files:
                csv_path = subject_dir / csv_fname
                label_fname = csv_fname.replace(".csv", ".label")
                label_path = subject_dir / label_fname

                if not label_path.exists():
                    continue

                ts_trans, _, lab_trans = self.extract_transition_segments(csv_path, label_path)
                ts_null, lab_null = self.extract_null_segments(csv_path, label_path)

                all_ts.extend(ts_trans + ts_null)
                all_labels.extend(lab_trans + lab_null)
                
        # Nếu đang load tập test và có classes đã biết từ tập train
        if known_classes is not None:
            valid_idx = [i for i, lab in enumerate(all_labels) if lab in known_classes]
            all_ts = [all_ts[i] for i in valid_idx]
            all_labels = [all_labels[i] for i in valid_idx]

        return np.array(all_ts), all_labels

    def preprocess(self, ts_array: np.ndarray):
        """
        Trích xuất thêm các features (VD: squared transformation) và chuẩn hóa.
        Input: (N, LENGTH, 3)
        Output: (N, 6, LENGTH) để train
        """
        ts_sq = np.square(ts_array)
        ts_combined = np.concatenate([ts_array, ts_sq], axis=2)  # (N, LENGTH, 6)

        # Min-max normalization
        datamin = np.min(ts_combined, axis=(1, 2), keepdims=True)
        datamax = np.max(ts_combined, axis=(1, 2), keepdims=True)
        denom = datamax - datamin
        denom[denom == 0] = 1e-8
        
        ts_norm = 2 * (ts_combined - datamin) / denom - 1
        
        # Chuyển kênh màu từ cuối lên đầu cho phù hợp với 1 số format
        return np.transpose(ts_norm, (0, 2, 1))  # (N, 6, LENGTH)

    def extract_sequence(self, subject: str, sequence_idx: int = 0):
        """Extract toàn bộ sequence cho detection method."""
        subject_dir = self.data_root / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {subject_dir}")
            
        csv_files = sorted([f for f in os.listdir(subject_dir) if f.startswith("HASC") and f.endswith(".csv")])
        if len(csv_files) <= sequence_idx:
            raise ValueError(f"Không có sequence index {sequence_idx} trong {subject_dir}")
            
        csv_fname = csv_files[sequence_idx]
        csv_path = subject_dir / csv_fname
        label_fname = csv_fname.replace(".csv", ".label")
        label_path = subject_dir / label_fname

        data = self.read_hasc_csv(csv_path)
        labels_df = self.read_hasc_label(label_path)

        t_start = labels_df["start"].min()
        t_end = labels_df["end"].max()
        mask = (data["time"] >= t_start) & (data["time"] <= t_end)
        seq = data[mask][["x", "y", "z"]].to_numpy()
        times = data[mask]["time"].to_numpy()

        return seq, times, labels_df, csv_fname
