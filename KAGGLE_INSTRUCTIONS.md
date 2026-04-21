# Hướng dẫn chạy model trên Kaggle

1. **Chuẩn bị Data**:  
   Dữ liệu mẫu (`SampleData20111104`) đã được cung cấp. Trên Kaggle, bạn có thể tạo một Dataset từ thư mục này.  
   *Lưu ý*: Script `config/config.py` đã cấu hình fallback thông minh:
   - Nếu tồn tại folder `/kaggle/input/...` (như khi mount trên Kaggle Workspace), nó sẽ đọc tự động.
   - Nếu chạy local, nó sẽ sử dụng đường dẫn `SampleData20111104` ở máy bạn.

2. **Cấu trúc Source Code**:
   Cấu trúc SOLID đã chia tách:
   - `config/`: Cấu hình siêu tham số (Epochs, Tốc độ học, Data Path,...).
   - `src/`: Bao gồm load dữ liệu (`data`), định nghĩa kiến trúc ResNet (`model`), huấn luyện (`training`), đánh giá (`evaluation`), và suy luận (`detection`).
   - `scripts/`: Chứa các entry-point để thực thi.
   - `outputs/`: Sẽ chứa tự động mô hình tốt nhất (`.keras`), logs, và figures (biểu đồ loss, confusion matrix,...).

3. **Chạy Trực tiếp trên Kaggle Notebook**:
   Tạo 1 Notebook mới, bật **GPU P100** hoặc **T4 x2**. Dán các lệnh sau vào cell:
   
   ```bash
   # Cài đặt thư viện
   !pip install -r requirements.txt
   
   # Chạy Training
   !python scripts/train.py
   
   # Đánh giá trên test subject
   !python scripts/evaluate.py
   
   # Vẽ biểu đồ detect trên sequence dài
   !python scripts/detect.py
   ```

   **Mách nhỏ**:  Kaggle có timeout 9/12h. Để train lâu (ví dụ đổi EPOCH trong `config.py` lên 200), chạy lệnh sau thông qua bash shell và tải kết quả `/kaggle/working/outputs/` về:
   ```bash
   !bash run_kaggle.sh
   ```
