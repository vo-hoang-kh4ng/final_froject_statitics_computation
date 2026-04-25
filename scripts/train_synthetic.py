import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.model.builder import MLPBuilder
from src.data_utils.stimulate_data import stimulate_data
import matplotlib.pyplot as plt

def main():
    config.setup()
    print("=" * 70)
    print("BƯỚC 1: HUẤN LUYỆN MLP TRÊN DỮ LIỆU TỔNG HỢP CAUCHY (SYNTHETIC DATA)")
    print("=" * 70)
    
    np.random.seed(config.NUMPY_SEED)
    tf.random.set_seed(config.TF_SEED)

    print("\n[PHASE 1] TẠO DỮ LIỆU GIẢ LẬP (CAUCHY DISTRIBUTION)")
    length_ts = config.WINDOW_LENGTH
    sample_size = 2000
    
    x_train, y_train, tau_train, mu_R_train = stimulate_data(
        length_ts=length_ts, 
        sample_size=sample_size, 
        scale=0.3, 
        ar_model_name='ARH'
    )
    
    print(f"[INFO] Tổng số mẫu: {len(x_train)}")
    
    # MLP 1D yêu cầu (N, Sequence, Channels)
    # Bỏ đi việc transpoes và concatenate 2 kênh vì MLP không cần tránh Pooling lỗi,
    # Nhưng giữ X^2 là phép biến đổi Variance hợp lý
    x_train = np.expand_dims(x_train, axis=-1)
    x_train_sq = np.square(x_train)
    x_train_cc = np.concatenate([x_train, x_train_sq], axis=-1) # (N, 100, 2)
        
    print(f"[INFO] Input shape: {x_train_cc.shape}")
    y_train = y_train.flatten()

    print("\n[PHASE 2] XÂY DỰNG MÔ HÌNH MLP")
    
    builder = MLPBuilder(
        n=length_ts,
        n_trans=x_train_cc.shape[-1], 
        n_layers=5,
        m_neurons=16,
        dropout_rate=config.DROPOUT_RATE,
        n_classes=2,
        model_name="Cauchy_MLP_5Layers"
    )
    model = builder.build()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\n[PHASE 3] HUẤN LUYỆN 200 EPOCHS")
    # Trainer không dùng ModelTrainer để loại bỏ cơ chế Early Stopping
    # Chạy thẳng với keras fit 200 epochs để hội tụ ở Cauchy
    
    log_csv_path = config.OUTPUT_DIR / "training_log_synthetic.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(str(log_csv_path))
    
    history = model.fit(
        x_train_cc, y_train,
        epochs=200,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[csv_logger],
        verbose=1
    )
    
    model.save(config.OUTPUT_DIR / "best_model_synthetic.keras")

    print("\n[PHASE 4] LƯU KẾT QUẢ VÀ LOGS")
    plot_path = config.FIGURES_DIR / "training_curves_synthetic_mlp.png"
    
    # Plot history bằng Matplotlib
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over 200 Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy over 200 Epochs')
    plt.tight_layout()
    plt.savefig(str(plot_path))
    plt.close()
    
    print(f"[SAVE] Biểu đồ training: {plot_path}")
    print("\nKết thúc huấn luyện Synthetic Data thành công!")

if __name__ == "__main__":
    main()
