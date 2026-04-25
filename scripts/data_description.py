import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Thêm root dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data import HascDataLoader
from src.data_utils.stimulate_data import stimulate_data

def describe_hasc_data():
    print("\n[INFO] ĐANG TRÍCH XUẤT MÔ TẢ DỮ LIỆU HASC...")
    loader = HascDataLoader(
        data_root=config.DATA_ROOT,
        length=config.WINDOW_LENGTH,
        size=config.EXTRACT_SIZE,
        size0=config.NULL_SIZE,
        num_trim=config.NUM_TRIM
    )
    
    # Lấy 1 chuỗi ngẫu nhiên để vẽ (Sequence đầu tiên)
    seq, times, labels_df, csv_fname = loader.extract_sequence(config.TEST_SUBJECT, sequence_idx=0)
    
    print(f"File mô tả: {csv_fname}")
    print(f"Tổng số bản ghi (Samples): {len(seq)}")
    
    # Bảng phân phối các trạng thái hành động
    print("\nBảng thống kê nhãn trạng thái (Transitions):")
    state_counts = labels_df['state'].value_counts()
    print(state_counts)
    
    # --- VẼ HÌNH MÔ TẢ CẢM BIẾN HASC ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, seq[:, 0], color="#2196F3", lw=1.0, alpha=0.8, label="X-axis (Lateral)")
    ax.plot(times, seq[:, 1], color="#4CAF50", lw=1.0, alpha=0.8, label="Y-axis (Vertical)")
    ax.plot(times, seq[:, 2], color="#FF9800", lw=1.0, alpha=0.8, label="Z-axis (Forward)")
    
    # Tô màu giới hạn các mốc thay đổi hành động
    for idx, row in labels_df.iterrows():
        mid_time = (row['start'] + row['end']) / 2
        ax.axvline(row['end'], color='gray', linestyle='--', alpha=0.5)
        ax.text(mid_time, ax.get_ylim()[1] * 0.9, row['state'], 
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    ax.set_title(f"Mô tả tín hiệu gia tốc kế phân giải cao - HASC ({csv_fname})", fontweight="bold")
    ax.set_xlabel("Thời gian (Time) giây")
    ax.set_ylabel("Gia tốc (Acceleration g)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = config.FIGURES_DIR / "hasc_data_description.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVE] Hình ảnh phân tích HASC: {save_path}")

def describe_synthetic_data():
    print("\n[INFO] ĐANG TRÍCH XUẤT MÔ TẢ DỮ LIỆU CAUCHY SYNTHETIC...")
    length_ts = config.WINDOW_LENGTH
    
    # Lấy phân phối Gaussian (chuẩn)
    print(" Sinh tập Gaussian (Control)...")
    x_gauss, y_gauss, tau_gauss, _ = stimulate_data(
        length_ts=length_ts, sample_size=10, scale=0.1, ar_model_name='Gaussian'
    )
    
    # Lấy phân phối Cauchy (Nhiễu đuôi dày)
    print(" Sinh tập Cauchy (Heavy-tail)...")
    x_cauchy, y_cauchy, tau_cauchy, _ = stimulate_data(
        length_ts=length_ts, sample_size=10, scale=0.3, ar_model_name='ARH'
    )
    
    # --- VẼ HÌNH SO SÁNH CAUCHY VS GAUSSIAN ---
    # Lấy 1 sequence rác có chứa Change-Point
    idx_g = np.where(y_gauss == 1)[0][0]
    idx_c = np.where(y_cauchy == 1)[0][0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # CUSUM Baseline (Gaussian)
    ax1.plot(x_gauss[idx_g], color="#3F51B5", lw=1.2)
    ax1.axvline(tau_gauss[idx_g], color="red", linestyle="--", label="Change-Point")
    ax1.set_title("Tín hiệu chuẩn (Gaussian Distribution) - Dễ phát hiện", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cauchy Heavy-tailed
    ax2.plot(x_cauchy[idx_c], color="#E91E63", lw=1.2)
    ax2.axvline(tau_cauchy[idx_c], color="red", linestyle="--", label="Change-Point")
    ax2.set_title("Tín hiệu nhiễu cực đoan (Cauchy Distribution) - Khó phát hiện, CUSUM thất bại", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = config.FIGURES_DIR / "synthetic_data_description.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"[SAVE] Hình ảnh phân tích Synthetic: {save_path}")

if __name__ == "__main__":
    config.setup()
    print("==================================================")
    print(" CHẠY PHÂN TÍCH VÀ MÔ TẢ DỮ LIỆU ĐỂ BÁO CÁO")
    print("==================================================")
    
    describe_hasc_data()
    describe_synthetic_data()
    
    print("\n[OK] Đã hoàn thành bộ mô tả dữ liệu!")
