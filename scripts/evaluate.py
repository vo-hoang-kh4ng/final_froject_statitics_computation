import os
import sys
import numpy as np

# Thêm root dự án vào sys.path để import được config và src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data import HascDataLoader
from src.evaluation import ModelEvaluator
from sklearn.preprocessing import LabelEncoder

def main():
    config.setup()
    print("=" * 70)
    print("BƯỚC 2: ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SUBJECT")
    print("=" * 70)
    
    np.random.seed(config.NUMPY_SEED)

    # 1. Load Data Setup
    print(f"\n[PHASE 1] TRÍCH XUẤT DỮ LIỆU TEST TỪ: {config.TEST_SUBJECT}")
    loader = HascDataLoader(
        data_root=config.DATA_ROOT,
        length=config.WINDOW_LENGTH,
        size=config.EXTRACT_SIZE,
        size0=config.NULL_SIZE,
        num_trim=config.NUM_TRIM
    )
    
    # Load Label Encoder để biết classes nào model đã học
    if not config.LABEL_ENCODER.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {config.LABEL_ENCODER}")
    
    le_classes = np.load(config.LABEL_ENCODER, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = le_classes
    
    # Chỉ load những nhãn đã biết
    ts_array, labels = loader.load_dataset([config.TEST_SUBJECT], known_classes=set(le.classes_))
    
    if len(ts_array) == 0:
        raise ValueError("Không trích xuất được file test hợp lệ.")
        
    x_test = loader.preprocess(ts_array)
    y_test = le.transform(labels)
    
    # 2. Evaluate
    print("\n[PHASE 2] ĐÁNH GIÁ MÔ HÌNH")
    evaluator = ModelEvaluator(
        model_path=config.BEST_MODEL,
        classes=le.classes_
    )
    
    results = evaluator.evaluate(x_test, y_test)
    
    print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    print("\nClassification Report:")
    print(results['report'])
    
    print("\n[PLOT] Vẽ Confusion Matrix...")
    cm_path = config.FIGURES_DIR / f"confusion_matrix_{config.TEST_SUBJECT}.png"
    evaluator.plot_confusion_matrix(
        y_test=y_test, 
        y_pred=results['y_pred'],
        title=f"AutoCPD - HASC: Confusion Matrix (Test: {config.TEST_SUBJECT})\nOverall Accuracy: {results['accuracy']*100:.1f}%",
        save_path=cm_path
    )
    print(f"[SAVE] Confusion matrix: {cm_path}")

    # 3. Binary Analysis
    print("\n[PHASE 3] PHÂN TÍCH BINARY: TRANSITION vs NULL")
    bin_results = evaluator.analyze_binary_detection(y_test, results['y_pred'])
    print(f"  True Positive (TP): {bin_results['tp']}")
    print(f"  True Negative (TN): {bin_results['tn']}")
    print(f"  False Positive (FP): {bin_results['fp']}")
    print(f"  False Negative (FN): {bin_results['fn']}")
    print(f"\n  Precision: {bin_results['precision']*100:.2f}%")
    print(f"  Recall (Detection Power): {bin_results['recall']*100:.2f}%")
    print(f"  F1-Score: {bin_results['f1']*100:.2f}%")
    print(f"  Binary Accuracy: {bin_results['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
