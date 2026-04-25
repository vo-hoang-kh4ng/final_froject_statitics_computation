import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.model.builder import MLPBuilder
from src.training import ModelTrainer
import autocpd.utils as ut
from src.data_utils.stimulate_data import stimulate_data

# Giảm xuống 2000 để chạy demo mượt mà trên máy tính. 
N_TRAIN = 2000  
N_TEST = 1000   

def gen_data_scenario(scenario_config, N_sub):
    x_data, y_data, _, _ = stimulate_data(
        length_ts=100, 
        sample_size=N_sub, 
        rho=scenario_config.get('ARcoef', 0),
        sigma=scenario_config.get('sigma', 1),
        scale=scenario_config.get('scale', 0.3),
        ar_model_name=scenario_config['ar_model']
    )
    y_data = y_data.flatten()
    return x_data, y_data

def evaluate_cusum(x_train, y_train, x_test, y_test):
    print("  [CUSUM] Tính toán CUSUM Statistic...")
    cusum_train = np.array([ut.MaxCUSUM(x) for x in x_train])
    cusum_test = np.array([ut.MaxCUSUM(x) for x in x_test])
    
    thresholds = np.linspace(np.min(cusum_train), np.max(cusum_train), 50)
    best_acc = 0
    best_th = 0
    for th in thresholds:
        preds = (cusum_train > th).astype(int)
        acc = accuracy_score(y_train, preds)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    preds_test = (cusum_test > best_th).astype(int)
    test_acc = accuracy_score(y_test, preds_test)
    return test_acc, best_th, preds_test

def evaluate_mlp(x_train, y_train, x_test, y_test, model_name):
    # MLP nhận Input (N, n, n_trans) = (N, 100, 2)
    x_tr_sq = np.square(x_train)
    x_tr_cc = np.stack([x_train, x_tr_sq], axis=-1)  # (N, 100, 2)
    
    x_te_sq = np.square(x_test)
    x_te_cc = np.stack([x_test, x_te_sq], axis=-1)

    # Dùng MLP với 5 lớp như Mentor chỉ định
    builder = MLPBuilder(
        n=100, n_trans=2, n_layers=5, m_neurons=32,
        dropout_rate=0.2, n_classes=2, model_name=model_name
    )
    model = builder.build()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("  [MLP] Bắt đầu Training 200 Epochs...")
    model.fit(x_tr_cc, y_train, epochs=200, batch_size=64, verbose=0, validation_split=0.1)
    
    probs = model.predict(x_te_cc, verbose=0)
    preds_test = np.argmax(probs, axis=1)
    test_acc = accuracy_score(y_test, preds_test)
    return test_acc, preds_test

def main():
    config.setup()
    np.random.seed(42)
    tf.random.set_seed(42)
    
    scenarios = {
        "S1": {'title': 'Gaussian iid, ρ=0', 'ar_model': 'Gaussian', 'sigma': 1},
        "S1_AR": {'title': 'Gaussian AR, ρ=0.7', 'ar_model': 'AR1', 'ARcoef': 0.7, 'sigma': 1},
        "S2": {'title': 'Gaussian Unif, ρ~U', 'ar_model': 'ARrho', 'sigma': 1.414},
        "S3": {'title': 'Cauchy Heavy-tail', 'ar_model': 'ARH', 'scale': 0.3}
    }
    
    results = {}
    
    # Khởi tạo khung vẽ Ma trận nhầm lẫn
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    fig.suptitle("COMPARISON OF CONFUSION MATRICES: CUSUM vs DEEP RESNET\n(Scoring Robustness against Noise and Outliers)", fontsize=18, fontweight='bold', y=0.95)
    
    col = 0
    for key, sconfig in scenarios.items():
        name = f"{key} ({sconfig['title']})"
        print(f"\n>>> ĐANG CHẠY KỊCH BẢN: {name}")
        
        # 1. Sinh Data
        x_train, y_train = gen_data_scenario(sconfig, N_sub=N_TRAIN)
        x_test, y_test = gen_data_scenario(sconfig, N_sub=N_TEST)
        
        # 2. Benchmark CUSUM
        cusum_acc, optimal_th, cusum_preds = evaluate_cusum(x_train, y_train, x_test, y_test)
        print(f"  [CUSUM] Độ chính xác: {cusum_acc*100:.2f}%")
        
        # 3. Benchmark MLP
        mlp_acc, mlp_preds = evaluate_mlp(x_train, y_train, x_test, y_test, model_name=f"Model_{key}")
        print(f"  [MLP] Độ chính xác: {mlp_acc*100:.2f}%")
        
        # PLOT CUSUM CONFUSION MATRIX (ROW 0)
        cm_cusum = confusion_matrix(y_test, cusum_preds)
        disp_cusum = ConfusionMatrixDisplay(confusion_matrix=cm_cusum, display_labels=['No CP', 'Change Point'])
        disp_cusum.plot(ax=axes[0, col], cmap='Blues', colorbar=False, values_format='d')
        axes[0, col].set_title(f"CUSUM Baseline\n{name}\nAcc: {cusum_acc*100:.1f}%", fontweight='bold', color='#1f77b4')
        axes[0, col].set_xlabel('Predicted Label')
        axes[0, col].set_ylabel('True Label')
        
        # PLOT MLP CONFUSION MATRIX (ROW 1)
        cm_mlp = confusion_matrix(y_test, mlp_preds)
        disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=['No CP', 'Change Point'])
        disp_mlp.plot(ax=axes[1, col], cmap='Oranges', colorbar=False, values_format='d')
        axes[1, col].set_title(f"AutoCPD MLP (5 Layers)\n{name}\nAcc: {mlp_acc*100:.1f}%", fontweight='bold', color='#d62728')
        axes[1, col].set_xlabel('Predicted Label')
        axes[1, col].set_ylabel('True Label')
        
        results[name] = {"CUSUM": cusum_acc, "MLP (5 Layers)": mlp_acc}
        col += 1
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.35)
    plot_path = config.FIGURES_DIR / "synthetic_confusion_matrices.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print(f"[SAVE] ĐÃ XUẤT ẢNH MA TRẬN NHẦM LẪN (HEATMAP) TẠI: {plot_path}")
    print("="*70)

if __name__ == "__main__":
    main()
