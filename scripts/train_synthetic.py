import os
import sys
import numpy as np
import tensorflow as tf

# Thêm root dự án vào sys.path để import được config và src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.model import ModelBuilder
from src.training import ModelTrainer
from src.data_utils.stimulate_data import stimulate_data

def main():
    config.setup()
    print("=" * 70)
    print("BƯỚC 1: HUẤN LUYỆN DEEP NN TRÊN DỮ LIỆU TỔNG HỢP CAUCHY (SYNTHETIC DATA)")
    print("=" * 70)
    
    np.random.seed(config.NUMPY_SEED)
    tf.random.set_seed(config.TF_SEED)

    # 1. Sinh dữ liệu ảo (Cauchy Distribution)
    # Cauchy distribution có tính chất Heavy-tailed (nhiễu ngoại lai giật chóp rất mạnh)
    # Đây là nơi CUSUM cổ điển chết hoàn toàn, nhưng Deep Learning sẽ tỏa sáng.
    print("\n[PHASE 1] TẠO DỮ LIỆU GIẢ LẬP (CAUCHY DISTRIBUTION)")
    length_ts = config.WINDOW_LENGTH
    sample_size = 2000 # Kích thước tập dữ liệu huấn luyện
    
    x_train, y_train, tau_train, mu_R_train = stimulate_data(
        length_ts=length_ts, 
        sample_size=sample_size, 
        scale=0.3, 
        ar_model_name='ARH' # Gọi model Cauchy heavy-tail
    )
    
    print(f"[INFO] Tổng số mẫu: {len(x_train)}")
    
    # Dữ liệu hiện đang là (N, length_ts). 
    # Mạng Conv2D bên trong general_deep_nn yêu cầu shape 3 chiều là (N, số kênh, thời gian)
    # Vì đây là chuỗi 1 chiều (1D Time Series) nên số kênh là 1.
    if len(x_train.shape) == 2:
        x_train = np.expand_dims(x_train, axis=1) # Trở thành (N, 1, length_ts)
    elif len(x_train.shape) == 3 and x_train.shape[-1] == 1:
        x_train = np.transpose(x_train, (0, 2, 1))
        
    # [FIX] Mạng general_deep_nn có lóp MaxPooling2D((2,2)) nên yêu cầu tối thiểu 2 kênh.
    # Ta ghép thêm 1 kênh dữ liệu bình phương (Squared) để x_train thành (N, 2, length_ts). 
    # Vừa chống crash lớp Pooling, vừa giúp ResNet học biến thiên theo bình phương (variance) siêu bén.
    x_train_sq = np.square(x_train)
    x_train = np.concatenate([x_train, x_train_sq], axis=1)
        
    print(f"[INFO] Input shape: {x_train.shape}")
    
    # Chuyển label mảng 2D thành vector 1D
    y_train = y_train.flatten()

    # 2. Xây dựng Model
    print("\n[PHASE 2] XÂY DỰNG MÔ HÌNH RESNET")
    # CỰC KỲ QUAN TRỌNG: Kernel_size phải đổi từ (3, 25) về (1, 25) 
    synthetic_kernel_size = (2, 25)
    
    builder = ModelBuilder(
        n=length_ts,
        n_trans=x_train.shape[1], # số channel = 1
        kernel_size=synthetic_kernel_size,
        n_filter=config.N_FILTER,
        dropout_rate=config.DROPOUT_RATE,
        n_classes=2, # Binary classification: 0 (Không điểm đổi) và 1 (Có điểm đổi)
        n_resblock=config.N_RESBLOCK,
        m=config.DENSE_WIDTHS,
        l=len(config.DENSE_WIDTHS),
        model_name="Cauchy_AutoCPD"
    )
    model = builder.build()
    model.summary()

    # 3. Training
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
    
    # Save vô file riêng để không đè mất kết quả HASC của bạn
    history = trainer.train(
        x_train=x_train,
        y_train=y_train,
        log_csv_path=config.OUTPUT_DIR / "training_log_synthetic.csv",
        best_model_path=config.OUTPUT_DIR / "best_model_synthetic.keras"
    )

    # 4. Xuất kết quả
    print("\n[PHASE 4] LƯU KẾT QUẢ VÀ LOGS")
    plot_path = config.FIGURES_DIR / "training_curves_synthetic.png"
    trainer.plot_history(history.history, plot_path)
    
    print(f"[SAVE] Biểu đồ training: {plot_path}")
    print("\nKết thúc huấn luyện Synthetic Data thành công!")

if __name__ == "__main__":
    main()
