import os
import sys
import numpy as np
import tensorflow as tf

# Thêm root dự án vào sys.path để import được config và src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data import HascDataLoader
from src.detection import ChangePointDetector

def main():
    config.setup()
    print("=" * 70)
    print("BƯỚC 3: PHÁT HIỆN CHANGE-POINTS TRÊN SEQUENCE THỰC")
    print("=" * 70)
    
    # 1. Load Data
    print(f"\n[PHASE 1] TRÍCH XUẤT SEQUENCE TỪ: {config.TEST_SUBJECT}")
    loader = HascDataLoader(
        data_root=config.DATA_ROOT,
        length=config.WINDOW_LENGTH,
        size=config.EXTRACT_SIZE,
        size0=config.NULL_SIZE,
        num_trim=config.NUM_TRIM
    )
    
    # Lấy sequence đầu tiên để detect
    seq, times, labels_df, csv_fname = loader.extract_sequence(config.TEST_SUBJECT, sequence_idx=0)
    print(f"[INFO] File: {csv_fname}")
    print(f"[INFO] Chiều dài sequence: {len(seq)} samples")
    
    # 2. Setup Detector
    print("\n[PHASE 2] SLIDING WINDOW INFERENCE")
    if not config.BEST_MODEL.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại {config.BEST_MODEL}")
    if not config.LABEL_ENCODER.exists():
        raise FileNotFoundError(f"Không tìm thấy encoder tại {config.LABEL_ENCODER}")
        
    model = tf.keras.models.load_model(str(config.BEST_MODEL))
    classes = np.load(config.LABEL_ENCODER, allow_pickle=True)
    
    detector = ChangePointDetector(model, classes, config)
    
    # 3. Detect
    cp_estimated_idx, p_smoothed, window_centers, probs, peaks = detector.detect(seq)
    
    true_cps_idx = detector.get_ground_truth_cps(labels_df, times, len(seq))
    
    print(f"\n[RESULTS] Ground truth có: {len(true_cps_idx)} CPs")
    print(f"[RESULTS] Phát hiện được : {len(cp_estimated_idx)} CPs")
    
    if len(cp_estimated_idx) > 0 and len(true_cps_idx) > 0:
        errors = []
        for true_cp in true_cps_idx:
            dists = np.abs(cp_estimated_idx - true_cp)
            nearest_est = cp_estimated_idx[np.argmin(dists)]
            error = abs(int(nearest_est) - int(true_cp))
            errors.append(error)

        threshold = config.WINDOW_LENGTH // 4
        n_detected = sum(1 for e in errors if e < threshold)
        detection_rate = n_detected / len(true_cps_idx) if len(true_cps_idx)>0 else 0
        
        print(f"\n  Mean detection error: {np.mean(errors):.1f} samples")
        print(f"  Detection Rate (error < {threshold}): {detection_rate*100:.1f}%")
    
    # 4. Plot
    print("\n[PHASE 3] VẼ KẾT QUẢ")
    plot_path = config.FIGURES_DIR / f"detection_{csv_fname.replace('.csv', '')}.png"
    detector.plot_detection(
        seq, times, labels_df, cp_estimated_idx, true_cps_idx, 
        p_transition=1 - probs[:, detector.null_indices].sum(axis=1) if len(detector.transition_indices)==0 else probs[:, detector.transition_indices].sum(axis=1),
        p_smoothed=p_smoothed, 
        window_centers=window_centers, 
        probs=probs, 
        peaks=peaks, 
        save_path=plot_path,
        title_info=f"Subject: {config.TEST_SUBJECT} | Seq: {csv_fname}"
    )
    print(f"[SAVE] Hình ảnh: {plot_path}")

if __name__ == "__main__":
    main()
