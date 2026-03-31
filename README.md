# 🇻🇳 Hệ Thống Đánh Giá Độ Tương Đồng Ngữ Nghĩa Thuật Ngữ Kỹ Thuật
### Khóa luận tốt nghiệp — Nguyễn Viết Huy & Hoàng Thanh Chiến

Hệ thống sử dụng các mô hình ngôn ngữ lớn (Transformers) để đo lường độ tương đồng ngữ nghĩa của thuật ngữ trong ngữ cảnh tiếng Việt. Công cụ hỗ trợ trích xuất vector, tính toán hệ số tương quan và phân tích thống kê chuyên sâu với giao diện trực quan.

---

## 🚀 ĐIỂM NỔI BẬT VỀ KỸ THUẬT

Hệ thống không chỉ dừng lại ở việc áp dụng mô hình có sẵn mà còn thực hiện các tối ưu hóa chuyên sâu:

* **Khắc phục hiện tượng Anisotropy (Tính dị hướng):** Đối với **PhoBERT-Large**, hệ thống áp dụng kỹ thuật **Last 4-Layer Averaging**. Thay vì chỉ lấy lớp cuối cùng (thường bị co cụm và mất tính phân biệt), chúng tôi trung bình cộng 4 lớp ẩn cuối để phổ điểm phân bố tự nhiên và chính xác hơn.
* **Xử lý dữ liệu lớn:** Giao diện được tối ưu để load toàn bộ dữ liệu (3000+ mẫu) mà vẫn đảm bảo độ mượt mà.
* **Giao diện thông minh:** Tự động co giãn (Rescale) các cột `Context` khi ẩn/hiện các cột điểm thành phần (`r1-r10`), giúp tối ưu không gian hiển thị.
* **Cơ chế Cache & Offline:** Tự động lưu trữ mô hình tại máy cục bộ sau lần chạy đầu tiên, cho phép chạy đánh giá hoàn toàn không cần kết nối Internet.

---

## 💻 YÊU CẦU HỆ THỐNG

| Thành phần     | Yêu cầu tối thiểu               |
|----------------|-------------------------------|
| Hệ điều hành   | Windows 10/11, Ubuntu 20.04+  |
| Python         | **3.10.11** (Bắt buộc)         |
| RAM            | 8 GB (Khuyến nghị 16 GB)      |
| Ổ cứng trống   | ~5 GB (Mô hình + Thư viện)     |

---

## 🛠 HƯỚNG DẪN CÀI ĐẶT (WINDOWS)

### 1. Cài đặt Python 3.10.11
Tải và cài đặt từ [Python.org](https://www.python.org/downloads/release/python-31011/). **Lưu ý: Tích chọn "Add Python to PATH"**.

### 2. Thiết lập môi trường ảo (Virtual Environment)
Mở Terminal tại thư mục dự án và chạy:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Lưu ý: Nếu máy có GPU NVIDIA, hãy cài đặt phiên bản Torch hỗ trợ CUDA để tăng tốc độ xử lý.*

---

## 📖 CƠ CHẾ ĐÁNH GIÁ (METHODOLOGY)

### Chiến lược trích xuất Vector (Embedding)
Mô hình trích xuất đặc trưng của từ mục tiêu dựa trên vị trí chính xác của nó trong câu ngữ cảnh:
* **PhoBERT-Base & mBERT:** Trích xuất từ lớp cuối cùng ($L_{last}$).
* **PhoBERT-Large:** Áp dụng công thức trung bình cộng 4 lớp cuối:
    $$v_{target} = \frac{1}{4} \sum_{i=21}^{24} Layer_i(token)$$

### Quy đổi điểm số
Điểm Cosine Similarity gốc (từ -1 đến 1) được quy đổi về thang điểm 10 (tương đương đánh giá của con người) theo phương pháp tuyến tính:
$$Scaled\_Score = \left( \frac{cos\_sim + 1}{2} \right) \times 10$$

---

## 🖥 HƯỚNG DẪN SỬ DỤNG

1.  **Khởi chạy:** Chạy lệnh `python app.py`.
2.  **Tải file:** Nhấn **"📂 Tải Dữ Liệu"** để chọn file Excel/CSV đầu vào.
3.  **Chọn Mô hình:** Lựa chọn giữa PhoBERT-Base, PhoBERT-Large hoặc mBERT.
4.  **Chạy Đánh Giá:** Nhấn **"▶ Chạy Đánh Giá"**. Kết quả Pearson và Spearman sẽ hiển thị ngay khi hoàn tất.
5.  **Phân tích Biểu đồ:** Nhấn **"📊 Xem Biểu Đồ"** để phân tích phân phối điểm và tương quan đồ thị.
6.  **Xuất Kết Quả:** Nhấn **"💾 Xuất Kết Quả"** để lưu lại file Excel đã có điểm đánh giá của mô hình.

---

## 📂 CẤU TRÚC THƯ MỤC

```text
semantic_eval/
├── core/
│   ├── models.py      # Định nghĩa mô hình & Logic trích xuất (Fix Anisotropy)
│   ├── evaluator.py   # Tính toán Cosine & Các hệ số tương quan
│   └── segmenter.py   # Bộ tách từ tiếng Việt (Underthesea/PyVi)
├── ui/
│   ├── app.py         # Giao diện chính (Tkinter)
│   └── plotting.py    # Logic vẽ biểu đồ thống kê
├── app.py             # File thực thi chính
└── requirements.txt   # Danh sách thư viện
```

---

## ⚠️ XỬ LÝ LỖI THƯỜNG GẶP

* **Lỗi mô hình không tải được:** Kiểm tra kết nối mạng lần đầu hoặc đặt biến môi trường `set HF_ENDPOINT=https://hf-mirror.com` nếu mạng bị chặn.
* **Lỗi hiển thị (Linux):** Nếu thiếu Tkinter, chạy: `sudo apt install python3.10-tk`.
* **Lỗi tràn bộ nhớ (OOM):** Nếu máy có RAM thấp, hãy ưu tiên sử dụng **PhoBERT-Base** thay vì bản Large.

---
*© 2024 - Khóa luận tốt nghiệp ĐH - Chuyên ngành Khoa học Máy tính.*