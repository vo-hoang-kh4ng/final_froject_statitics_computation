import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Thêm root dự án vào sys.path để import được config và src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data import HascDataLoader
from src.model import ModelBuilder
from src.training import ModelTrainer

def main():
    config.setup()
    print("=" * 70)
    print("BƯỚC 1: HUẤN LUYỆN DEEP NEURAL NETWORK TRÊN DỮ LIỆU HASC")
    print("=" * 70)
    
    # Random seeds
    np.random.seed(config.NUMPY_SEED)
    tf.random.set_seed(config.TF_SEED)

    # 1. Load & Extract Data
    print("\n[PHASE 1] TRÍCH XUẤT VÀ TIỀN XỬ LÝ DỮ LIỆU")
    loader = HascDataLoader(
        data_root=config.DATA_ROOT,
        length=config.WINDOW_LENGTH,
        size=config.EXTRACT_SIZE,
        size0=config.NULL_SIZE,
        num_trim=config.NUM_TRIM
    )
    
    ts_array, labels = loader.load_dataset(config.TRAIN_SUBJECTS)
    if len(ts_array) == 0:
        raise ValueError("Không trích xuất được dữ liệu! Vui lòng kiểm tra lại DATA_ROOT.")
        
    print(f"[INFO] Tổng số mẫu: {len(ts_array)}")
    
    # 2. Preprocess Data
    x_data = loader.preprocess(ts_array)
    print(f"[INFO] Input shape: {x_data.shape}")

    # 3. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"[INFO] Số lớp: {num_classes}")
    
    # Lưu encoder
    np.save(config.LABEL_ENCODER, le.classes_)
    print(f"[SAVE] Label encoder: {config.LABEL_ENCODER}")

    # Shuffle dataset
    idx = np.random.permutation(x_data.shape[0])
    x_data = x_data[idx]
    y_data = y_encoded[idx]

    # 4. Build Model
    print("\n[PHASE 2] XÂY DỰNG MÔ HÌNH")
    builder = ModelBuilder(
        n=config.WINDOW_LENGTH,
        n_trans=x_data.shape[1],
        kernel_size=config.KERNEL_SIZE,
        n_filter=config.N_FILTER,
        dropout_rate=config.DROPOUT_RATE,
        n_classes=num_classes,
        n_resblock=config.N_RESBLOCK,
        m=config.DENSE_WIDTHS,
        l=len(config.DENSE_WIDTHS),
        model_name=config.MODEL_NAME
    )
    model = builder.build()
    model.summary()

    # 5. Train Model
    print("\n[PHASE 3] HUẤN LUYỆN")
    trainer = ModelTrainer(
        model=model,
        learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_RATE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        early_stop_patience=config.EARLY_STOP_PATIENCE
    )
    trainer.compile()
    
    history = trainer.train(
        x_train=x_data,
        y_train=y_data,
        log_csv_path=config.LOG_CSV,
        best_model_path=config.BEST_MODEL
    )

    # 6. Save & Evaluate
    print("\n[PHASE 4] LƯU KẾT QUẢ VÀ LOGS")
    np.save(config.HISTORY_FILE, history.history)
    print(f"[SAVE] History: {config.HISTORY_FILE}")
    
    plot_path = config.FIGURES_DIR / "training_curves.png"
    trainer.plot_history(history.history, plot_path)
    print(f"[SAVE] Biểu đồ training: {plot_path}")

    print("\nKết thúc huấn luyện!")

if __name__ == "__main__":
    main()
