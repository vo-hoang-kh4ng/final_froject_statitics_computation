# AutoCPD - Time Series Change-Point Detection via Deep Learning

Dự án này là mã nguồn triển khai mô hình học sâu **ResNet** để thực thi bài toán phát hiện điểm thay đổi (Change-Point Detection) trên dữ liệu chuỗi thời gian (cụ thể là dữ liệu gia tốc kế của HASC), lấy cảm hứng từ cấu trúc thuật toán [AutoCPD](https://github.com/Jieli12/AutoCPD) (Li et al., 2023). 

Mã nguồn đã được refactor lại theo tiêu chuẩn lập trình **SOLID** nhằm giúp việc mở rộng, thêm dữ liệu, và đẩy lên các nền tảng huấn luyện lớn (như **Kaggle / Google Colab**) trở nên dễ dàng nhất.

---

## 📂 Tổ Chức Thư Mục

```text
├── config/
│   └── config.py              # Đóng vai trò là "Single source of truth" (siêu tham số, đường dẫn)
├── src/
│   ├── data/hasc_loader.py    # D (Data): Đọc file CSV, extract window transition/null
│   ├── detection/detector.py  # D (Detection): Thuật toán xử lý trượt (sliding window)
│   ├── evaluation/evaluator.py# E (Eval): Đánh giá F1-score, vẽ Confusion Matrix
│   ├── model/builder.py       # M (Model): Định nghĩa cấu trúc Deep_ResNet 
│   └── training/trainer.py    # T (Training): Compile model, Callbacks (Early stopping)
├── scripts/
│   ├── train.py               # (Kaggle Step 1) - Huấn luyện mô hình từ đầu
│   ├── evaluate.py            # (Kaggle Step 2) - Test trên Subject mới (ví dụ person107)
│   └── detect.py              # (Kaggle Step 3) - Visualize điểm thay đổi trên chuỗi thời gian thực
├── run_kaggle.sh              # 🚀 Shell command setup tự động hóa luồng chạy cho Kaggle/Server
├── KAGGLE_INSTRUCTIONS.md     # 📖 File hướng dẫn cấu hình nền tảng Kaggle
└── requirements.txt           # Danh sách các thư viện pip cần cài đặt
```

---

## 🚀 Hướng Dẫn Nhanh Chạy Trên Server (Local / Kaggle)

Hệ thống sẽ không push dữ liệu thô `SampleData20111104` lên Git để tránh làm nặng Repo. Bạn cần đưa dữ liệu này về local / upload nó thành dataset trên Kaggle.

1. **Khởi tạo môi trường & cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy kịch bản tự động**:
   ```bash
   bash run_kaggle.sh
   # Bash file này sẽ đóng gói chạy theo trình tự: Train -> Eval -> Detect
   ```

3. **Kiểm tra kết quả sinh ra**:
   Toàn bộ file trọng số mô hình `.keras`, file logs huấn luyện, cũng như file hình ảnh `.png` confusion matrix/đường loss biểu đồ sẽ đều được tự động kết xuất ra mục `outputs/`.