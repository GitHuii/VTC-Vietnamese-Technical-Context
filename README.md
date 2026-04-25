# 🇻🇳 Hệ Thống Đánh Giá Độ Tương Đồng Ngữ Nghĩa Thuật Ngữ Kỹ Thuật
### Khóa luận tốt nghiệp — Nguyễn Viết Huy & Hoàng Thanh Chiến

Hệ thống sử dụng các mô hình ngôn ngữ lớn (Transformers) để đo lường độ tương đồng ngữ nghĩa của thuật ngữ trong ngữ cảnh tiếng Việt. Công cụ hỗ trợ trích xuất vector, tính toán hệ số tương quan (Pearson/Spearman) và phân tích thống kê chuyên sâu.

---

## 💻 YÊU CẦU HỆ THỐNG & CÀI ĐẶT

| Thành phần     | Yêu cầu tối thiểu               |
|----------------|-------------------------------|
| Hệ điều hành   | Windows 10/11, Ubuntu 20.04+  |
| Python         | **3.10.11** (Bắt buộc)         |
| GPU (Khuyên dùng)| NVIDIA hỗ trợ CUDA 11.8+      |

**Cài đặt nhanh (Terminal):**
```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate  # (Windows)
# source venv/bin/activate (Linux/Mac)

# 2. Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

*(Lưu ý: Để hệ thống nhận diện và chạy bằng GPU, hãy đảm bảo bạn đã cài đặt phiên bản PyTorch hỗ trợ CUDA phù hợp với máy của bạn).*

---

## 📖 CÁC MÔ HÌNH HỖ TRỢ & CHIẾN LƯỢC VECTOR

Hệ thống hiện tại tích hợp 3 mô hình ngôn ngữ phổ biến nhằm mục đích thực nghiệm và đối chiếu chéo:

1. **PhoBERT-Base (`vinai/phobert-base`):** Trích xuất vector từ lớp cuối cùng (Last hidden state). Phân bố điểm đồng đều, thích hợp cho tiếng Việt.
2. **mBERT (`bert-base-multilingual-cased`):** Mô hình đa ngôn ngữ cơ sở của Google.
3. **XLM-RoBERTa-Base (`xlm-roberta-base`):** Tính toán trung bình cộng 4 lớp cuối (Last 4-layer averaging). Đây là mô hình phục vụ báo cáo thực nghiệm về hiện tượng phân bố điểm co cụm trên các kiến trúc RoBERTa-based.

**Công thức quy đổi điểm:**
$$Score = \max(0, \cos(\vec{u}, \vec{v}))$$

---

## 🖥 HƯỚNG DẪN SỬ DỤNG

1. **Khởi chạy:** Mở terminal, gõ `python app.py` (Giao diện Tkinter sẽ xuất hiện).
2. **Tải file:** Nhấn **"📂 Tải Dữ Liệu"** chọn file Excel/CSV.
3. **Chọn mô hình:** Chọn 1 trong 3 mô hình từ Dropdown list.
4. **Chạy đánh giá:** Nhấn **"▶ Chạy Đánh Giá"**. Theo dõi log bên dưới để biết mô hình đang chạy trên thiết bị nào (`cpu` hay `cuda`).
5. **Đồ thị:** Nhấn **"📊 Xem Biểu Đồ"** để thấy rõ sự khác biệt phân bố (co cụm vs trải rộng) giữa các mô hình.

---
*© 2026 - Khóa luận tốt nghiệp Đại hokc Kinh Tế - Kỹ Thuật Công Nghiệp*
```