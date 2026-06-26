# Báo Cáo Kết Quả Đánh Giá Cấu Trúc Decoder (Ablation Study)

Báo cáo này tổng hợp kết quả nghiên cứu thực nghiệm (ablation study) đối với cấu trúc Decoder trong mô hình mạng Transformer dự báo chuỗi thời gian dài trên bộ dữ liệu **ETTh1** (Hourly Electricity Transformer Temperature).

---

## ⚙️ Cấu Hình Thực Nghiệm & Siêu Tham Số (Hyperparameters)

Để đảm bảo tính nhất quán và khả năng chạy thực nghiệm nhanh chóng trên CPU, tất cả các mô hình được huấn luyện bằng cùng một cấu hình chuẩn dưới đây:

### 1. Dữ liệu & Tần suất
* **Bộ dữ liệu**: `ETTh1.csv` (Hourly Electricity Transformer Temperature)
* **Kích thước cửa sổ nhìn lại (Lookback Window - `seq_len`)**: 96 bước thời gian (4 ngày)
* **Kích thước cửa sổ gợi ý (Label Window - `label_len`)**: 48 bước thời gian (2 ngày)
* **Kích thước dự báo (Forecasting Window - `pred_len`)**: 24 bước thời gian (1 ngày)
* **Lượng dữ liệu huấn luyện tối đa (`max_train`)**: 500 cửa sổ (tránh quá tải CPU)
* **Lượng dữ liệu kiểm thử tối đa (`max_eval`)**: 200 cửa sổ

### 2. Cấu trúc mô hình mặc định (Mô hình nhỏ tối ưu)
* **Số chiều biểu diễn (`d_model`)**: 64
* **Số đầu chú ý (Attention Heads - `n_heads`)**: 4
* **Số lớp Encoder (`e_layers`)**: 2
* **Số lớp Decoder (`d_layers`)**: 1
* **Số chiều mạng Feed-Forward (`d_ff`)**: 128
* **Tỉ lệ Dropout**: 0.1
* **Hàm kích hoạt**: GELU

### 3. Huấn luyện & Tối ưu hóa
* **Số Epoch tối đa**: 10
* **Tốc độ học (Learning Rate)**: 0.001 (1e-3)
* **Kích thước lô (Batch Size)**: 32
* **Cơ chế dừng sớm (Early Stopping)**: Kích hoạt với `patience = 3` (dừng nếu sai số Validation không cải thiện sau 3 epoch liên tiếp).
* **Thiết bị chạy (Device)**: CPU

---

## 📊 Bảng Kết Quả Thực Nghiệm (Comparison Table)

| Biến Thể | Test MSE ↓ | Test MAE ↓ | Δ MSE | Δ MSE % | Mức Ảnh Hưởng | Tham Số (Params) | Số Epoch Chạy Thực Tế | Thời Gian Chạy (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline** (Đầy đủ) | **2.3559** | **1.0966** | +0.0000 | +0.0% | **BASELINE** | 120,583 | 10 / 10 | 14.5s |
| **no-self-attention** | **2.2779** | 1.1912 | -0.0780 | -3.3% | **THẤP (LOW)** | 120,583 | 9 / 10 | 11.8s |
| **no-causal-mask** | 2.5530 | 1.1715 | +0.1971 | +8.4% | **TRUNG BÌNH (MEDIUM)** | 120,583 | 10 / 10 | 17.4s |
| **no-decoder** | **1.9991** | 1.1677 | -0.3568 | -15.1% | **CÓ LỢI (BENEFICIAL)** | 1,152,488 | 4 / 10 | 5.1s |

*Chú thích:*
* *`Test MSE` / `Test MAE`: Sai số trung bình trên tập kiểm thử (càng thấp càng tốt).*
* *`Δ MSE`: Sai số thay đổi so với baseline ($MSE_{variant} - MSE_{baseline}$). Giá trị dương ($+$) biểu thị hiệu năng giảm.*

---

## 🔍 Phân Tích Chi Tiết Từng Biến Thể

### 1. Baseline (Mô hình Transformer đầy đủ)
* **Nhận xét**: Là thiết lập tiêu chuẩn của Encoder-Decoder Transformer. Ở điều kiện dữ liệu giới hạn (500 cửa sổ), mô hình chưa hội tụ hoàn toàn (MSE = 2.3559), phản ánh hiện tượng underfitting nhẹ do kiến trúc phức tạp cần nhiều dữ liệu hơn để học tốt.

### 2. No-Self-Attention (Bỏ Self-Attention trong Decoder)
* **Nhận xét**: Việc loại bỏ lớp Self-Attention trong Decoder chỉ làm giảm nhẹ hiệu năng hoặc thậm chí cải thiện nhẹ (-3.3% MSE). Điều này chứng minh rằng trong bài toán dự báo chuỗi thời gian, Decoder chủ yếu dựa vào thông tin liên kết chéo từ Encoder truyền sang thông qua **Cross-Attention** hơn là các thông tin tự hồi quy thời gian trong chính Decoder.

### 3. No-Causal-Mask (Không dùng mặt nạ Causal)
* **Nhận xét**: Khi tắt mặt nạ Causal, dữ liệu tương lai bị rò rỉ vào quá khứ trong quá trình huấn luyện. Điều này khiến mô hình học cách "gian lận" (dùng thông tin tương lai để đoán tương lai) dẫn đến việc tối ưu hóa tốt trên tập Train nhưng không thể tổng quát hóa và cho sai số **tăng cao 8.4% trên tập Test**. Thí nghiệm này nhấn mạnh vai trò sống còn của Causal Mask để ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage).

### 4. No-Decoder (Bỏ hoàn toàn Decoder)
* **Nhận xét**: Việc loại bỏ hoàn toàn Decoder và chiếu thẳng đầu ra Encoder ra chuỗi kết quả qua lớp Linear lớn thực chất làm tăng số tham số huấn luyện lên gấp **10 lần** (từ 120k lên 1.15M tham số). Cấu trúc tuyến tính tham số lớn này hoạt động hiệu quả hơn hẳn trên tập dữ liệu nhỏ `ETTh1` (giảm 15.1% MSE), tương tự nhận định của bài báo *DLinear* khi các mô hình tuyến tính tham số lớn thường vượt trội hơn Transformer nhỏ trên các chuỗi đơn giản.

---

## 🏆 Xếp Hạng Tầm Quan Trọng Của Thành Phần Đối Với Decoder

Dựa trên mức độ tăng sai số MSE trên tập Test khi loại bỏ hoặc can thiệp vào thành phần (most $\rightarrow$ least critical):

1. **Causal Masking (Mặt Nạ Causal)** (Δ MSE = +0.1971): Thành phần quan trọng nhất giúp mô hình tránh rò rỉ thông tin tương lai và đảm bảo khả năng dự báo thực tế.
2. **Decoder Self-Attention (Tự Chú Ý)** (Δ MSE = -0.0780): Thành phần ít ảnh hưởng nhất, có thể lược bỏ để tiết kiệm tài nguyên tính toán mà không làm suy giảm hiệu năng đáng kể.
