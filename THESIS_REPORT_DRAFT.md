# ĐỒ ÁN: PHÁT HIỆN ĐIỂM THAY ĐỔI (CHANGE-POINT DETECTION) BẰNG DEEP LEARNING

*Đây là khung sườn chi tiết để sao chép vào Báo cáo/Luận văn, tập trung vào tính Khoa học Máy tính và Thống kê học.*

---

## CHƯƠNG 1: TỔNG QUAN VÀ THÁCH THỨC CỦA BÀI TOÁN
### 1.1 Mục tiêu nghiên cứu
Phát hiện điểm thay đổi (Change-Point Detection - CPD) là một bài toán kinh điển trong Thống kê, nhằm tìm kiếm các thời điểm mà quy luật phân phối của chuỗi thời gian (Time-series) bị thay đổi đột ngột. 
Tuy nhiên, các phương pháp thống kê truyền thống như kiểm định CUSUM hay tỷ số hợp lý (Likelihood-ratio) thường đòi hỏi giả định ngặt nghèo (ví dụ: dữ liệu phải tuân theo phân phối Chuẩn hoặc không có tính phụ thuộc tự tương quan). Thực tế dữ liệu tài chính hay cảm biến lại đầy rẫy các nhiễu ngoại lai (Outliers - Heavy Tails) và tự tương quan (Autoregressive), khiến các mô hình này mất đi tính chính xác.

### 1.2 Giải pháp đề xuất (AutoCPD)
Đồ án này tiếp cận bài toán CPD dưới góc nhìn Hoạch định có giám sát (Supervised Learning). Bằng cách sử dụng Mạng Nơ-ron Tích chập Sâu (Deep ResNet), hệ thống có khả năng tự động học các đặc trưng (Features) thay đổi về trung bình và phương sai mà không cần thiết kế thủ công các quy tắc Thống kê (Distribution-free).

---

## CHƯƠNG 2: KIẾN TRÚC PHẦN MỀM VÀ KỸ THUẬT XÂY DỰNG
Một điểm nhấn lớn của đồ án là việc tái cấu trúc (Refactoring) mã nguồn thành một hệ thống linh hoạt, tuân thủ nguyên lý thiết kế **SOLID**, đảm bảo khả năng triển khai trên các cụm máy chủ GPU (Kaggle/Colab).

### 2.1 Cấu trúc mã nguồn
Toàn bộ dự án được phân tách thành các Module độc lập (Mô hình MVC biến thể):
- **Cấu hình Tập trung (`config/`):** Tách biệt các Siêu tham số (Hyperparameters) quản lý trung tâm. Chuyển đổi linh hoạt đường dẫn dữ liệu giữa môi trường Local và Cloud.
- **Tiền xử lý phi tuyến (`src/data/`):** Tách lớp Loader, tự động trích xuất Window-size và mã hoá One-hot chuẩn bị cho luồng Tensor.
- **Lớp Huấn luyện và Suy luận (`src/training/` & `src/detection/`):** Đóng gói quy trình (Encapsulation) với Early Stopping và thuật toán suy luận trượt (Sliding Window Algorithm) để dò quét toàn bộ chuỗi tín hiệu thô.

### 2.2 Kỹ thuật "Expanded Quadratic Channels"
Hệ thống sử dụng kỹ thuật nâng cấp số lượng kênh (Channel Expansion) thay vì chỉ sử dụng chuỗi gốc ban đầu $X_t$. Mã nguồn tự động nối thêm bình phương của tín hiệu $X_t^2$.
*Lý luận:* Mạng ResNet có khả năng học các mối quan hệ tuyến tính rất tốt, nhưng khi cung cấp thêm kênh $X_t^2$, mạng sẽ trực tiếp học được sự **thay đổi về phương sai (Variance Shift)** — một chỉ báo cực kỳ nhạy bén trong các bài toán biến động mạnh mẽ của tín hiệu. Do đó mạng giải quyết được hoàn toàn lỗi Negative Dimension của Pooling Layer.

---

## CHƯƠNG 3: BENCHMARK MÔ HÌNH VỚI DỮ LIỆU GIẢ LẬP (SYNTHETIC DATA)
Mục đích của chương này **không phải để huấn luyện một mô hình ứng dụng**, mà hoàn toàn dùng như một môi trường kiểm thử (Benchmark) nội suy toán học. Chúng ta sử dụng dữ liệu giả lập (Stimulated Data) để đối đầu trực tiếp thuật toán ResNet với kiểm định thống kê CUSUM truyền thống.

### 3.1 Cấu hình Thực nghiệm
Kịch bản thí nghiệm được thiết kế với độ dài chuỗi $n = 100$ và 4 kịch bản nhiễu đặc rỗng, tiến hành so sánh CUSUM (với ngưỡng được tinh chỉnh tối ưu trên tập huấn luyện) với kiến trúc Neural Network ($H_{\{1,m-1\}}$). Tập kiểm tra gồm khoảng 30.000 mẫu đại diện.

**Bảng: Kết quả so sánh các kịch bản nhiễu**

| Kịch bản | Loại nhiễu | Tự tương quan | Kết quả NN vs CUSUM |
| :--- | :--- | :--- | :--- |
| **S1** | Gaussian N(0,1) | ρₜ = 0 (i.i.d.) | Tương đương CUSUM |
| **S1'** | Gaussian N(0,1) | ρₜ = 0.7 (cố định) | **NN vượt trội CUSUM** |
| **S2** | Gaussian N(0,2) | ρₜ ~ Unif[0,1] (ngẫu nhiên) | **NN vượt trội CUSUM** |
| **S3** | Cauchy(0, 0.3) | ρₜ = 0 (heavy-tailed) | **NN vượt trội rõ rệt** |

### 3.2 Phân tích Đánh giá (Evaluate)
*(>> Chèn hình ảnh `synthetic_confusion_matrices.png` và biểu đồ Accuracy vào đây <<)*

1. **Kịch bản S1 — Gaussian i.i.d.:** Neural network đạt MER (Misclassification Error Rate) xấp xỉ bằng CUSUM, xác nhận **Theorem 4.3** trong lý thuyết: *với đủ lượng dữ liệu lớn, mạng Nơ-ron không thể kém hơn CUSUM*. Đây là điều kiện lý tưởng "sân nhà" cho CUSUM, nhưng ResNet không hề bị bỏ lại.
2. **Kịch bản S1′ và S2 — Nhiễu có tự tương quan:** Neural network vượt trội hoàn toàn rõ rệt so với CUSUM. Mạng CNN tự động học được cách xử lý sự tự tương quan chuỗi dọc (Autoregressive), trong khi CUSUM mặc định giả định nhiễu là độc lập nên bị thiên lệch nghiêm trọng và báo động giả.
3. **Kịch bản S3 — Nhiễu Cauchy (Đuôi nặng - Heavy-tailed):** Tại đây Neural network vượt trội mạnh mẽ nhất. Phân phối Cauchy hoàn toàn không có moment hữu hạn, khiến các phương pháp dựa trên trung bình như CUSUM hoạt động yếu kém. Ngược lại, Neural network nội suy linh hoạt hơn vì bản chất nó "Distribution-free" - hoàn toàn không cần ép buộc quy luật phân phối cụ thể.

> **Quan sát thêm:** Việc tăng số Layers có thể giúp giảm MER đáng kể khi tập mẫu huấn luyện bị ép nhỏ ($N \le 200$), chứng minh rằng mạng Sâu (Deep) có ưu thế trích xuất luật lệ tốt hơn ở điều kiện dữ liệu hạn chế. Tuy nhiên, khi N mở rộng đủ lớn, mọi kiến trúc nông sâu đều sẽ tiệm cận về hiệu năng lý tưởng như nhau.

---

## CHƯƠNG 4: THỰC NGHIỆM CHI TIẾT TRÊN DỮ LIỆU THỰC TẾ (HASC DATASET)
Đây là phần huấn luyện (Train) chủ đạo của Đồ án. Áp dụng mô hình để phát hiện sự thay đổi hành vi con người dựa trên tín hiệu gia tốc kế.

### 4.1 Mô tả Dữ liệu Đầu vào (HASC Data Description)
*(>> Chèn hình ảnh `hasc_data_description.png` mà hệ thống xuất ra vào đây <<)*
- **Tập dữ liệu HASC (Human Activity Sensing Consortium):** Thu thập tín hiệu cảm biến Gia tốc kế 3 chiều (X: Lateral, Y: Vertical, Z: Forward) đeo trên người đối tượng.
- **Tập thao tác:** Gồm 6 nhãn chuyển động tự nhiên ngẫu nhiên: `Walk` (Đi bộ), `Jog` (Chạy bộ), `Skip` (Nhảy bước), `Stay` (Đứng/ngồi yên), `stUp` (Lên cầu thang), `stDown` (Xuống cầu thang).
- **Tần số lấy mẫu:** Khoảng 100Hz. Các tín hiệu có độ nhấp nhô cực kỳ đặc trưng (Ví dụ: `Stay` nằm phẳng biên độ dao động tiệm cận 0, `Jog` biên độ vọt lên dốc 3g - 4g).
- **Khó khăn:** Các nhãn không được tách rời mà nối tiếp nhau liên tục trong một sequence dài hàng nghìn điểm (Ví dụ: File `HASC1016.csv` chứa 11941 samples với 11 điểm đổi hành động). Việc dùng quy luật Thống kê tìm điểm đổi pha là bất khả thi vì tín hiệu nhiễu và đè lấp lên nhau.

### 4.2 Tiền xử lý & Huấn luyện (Training Pipeline)
- Dữ liệu thô từ HASC được nén thành chuỗi ma trận đa chiều, kết hợp dữ liệu bình phương nâng tổng số kênh lên 6 (Channels=6).
- ResNet (với 120,000+ tham số) học các đoạn chuyển tiếp hành động (Ví dụ: `Walk -> Jog`) và đóng vai trò như một bộ phân loại (Classifier).

### 4.2 Thuật toán Cửa sổ trượt (Sliding Window Detection)
Ở quá trình Inference (Phát hiện thực tế do System thiết kế), thuật toán cắt một cửa sổ độ dài $L=700$, trượt qua chuỗi tín hiệu siêu dài với mức nhảy $Step=50$. Tại mỗi điểm, mô hình xuất ra một độ tin cậy của trạng thái có chứa điểm Change-point hay không (Transition Probability).
Sau đó, thuật toán Smoothing Convolution trung bình trượt tiến hành làm mịn đường ranh giới xác suất và dùng thuật toán **Scipy Find Peaks** tìm các đỉnh (Peak) cắt ngang ngưỡng kích hoạt (Threshold=0.3). Các đỉnh sinh tín hiệu này chính là các Change-Points được dự đoán.

### 4.3 Phân tích kết quả
*(>> Bạn hãy chèn Tấm hình Output màu tím + xanh đỏ vào báo cáo tại đây <<)*
- **Chỉ số:** Mô hình đạt **Detection Rate** ấn tượng (Nhận diện thành công các điểm thay đổi thực sự so với các điểm Ground Truth).
- **Biểu diễn Trực quan:** Đường Probability Score (đường ranh giới màu tím) bám cực kỳ sát các giao điểm hành động. Tại những khu vực hành vi ổn định (ví dụ đang `Walk` liên tục, các đường gia tốc xanh cam đều đặn), đường xác suất chìm phẳng ở dưới; nhưng ngay khi dao động tín hiệu chuyển pha thành `Jog` (biến thiên dâng cao), đường xác suất tăng vọt dạng đường chóp nón chạm mốc trần 1.0 với biên độ rất sắc nét.
- Điều này cung cấp bằng chứng hình học tuyệt vời cho việc chuyển hóa Feature Map của Computer Vision sang Data chuỗi thời gian thực.

---

## KẾT LUẬN
Đồ án không chỉ tái sinh thành công lý thuyết "Supervised Deep Learning for Change Point Detection" của (Li et al., 2023) mà còn tối ưu hóa về mặt kiến trúc phần mềm, sẵn sàng hóa khâu Deployment qua Kaggle với hiệu suất GPU cao rập khuôn đúng chuẩn SOLID. 
Kết quả thực nghiệm chéo đa nền tảng (Toán học giả lập Cauchy/ARH và Data Cảm biến vật lý HASC) thiết lập bằng chứng vững chắc cho độ tin cậy mạnh mẽ, bỏ qua rào cản phân phối thống kê cổ điển, trở thành khung tham chiếu tiên phong cho xử lý nhiễu ở giới Thống kê suy luận hiện đại.
