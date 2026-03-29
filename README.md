# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY HỆ THỐNG
## Hệ thống đánh giá độ tương đồng ngữ nghĩa thuật ngữ kỹ thuật
### Khóa luận tốt nghiệp — Nguyễn Viết Huy & Hoàng Thanh Chiến

---

## YÊU CẦU HỆ THỐNG

| Thành phần     | Yêu cầu tối thiểu              |
|----------------|-------------------------------|
| Hệ điều hành   | Windows 10/11, Ubuntu 20.04+  |
| Python         | **3.10.11** (bắt buộc)         |
| RAM            | 8 GB (khuyến nghị 16 GB)      |
| Ổ cứng trống   | ~5 GB (mô hình + thư viện)    |
| GPU (tùy chọn) | NVIDIA CUDA 11.8+             |

---

## BƯỚC 1 — CÀI ĐẶT PYTHON 3.10.11

### Windows
1. Truy cập: https://www.python.org/downloads/release/python-31011/
2. Tải file `python-3.10.11-amd64.exe`
3. Chạy installer, **tích chọn "Add Python to PATH"**
4. Xác nhận cài đặt thành công:
   ```
   python --version
   # Kết quả: Python 3.10.11
   ```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-venv python3.10-tk -y

# Xác nhận
python3.10 --version
```

> **Lưu ý Ubuntu:** Tkinter cần được cài riêng với lệnh trên (`python3.10-tk`).
> Nếu thiếu, giao diện sẽ không khởi động được.

---

## BƯỚC 2 — TẠO MÔI TRƯỜNG ẢO (VIRTUAL ENVIRONMENT)

Sử dụng môi trường ảo để tránh xung đột thư viện với các dự án khác.

### Windows (Command Prompt hoặc PowerShell)
```cmd
:: Di chuyển vào thư mục dự án
cd C:\path\to\semantic_eval

:: Tạo môi trường ảo tên "venv"
python -m venv venv

:: Kích hoạt môi trường ảo
venv\Scripts\activate

:: Dấu nhắc lệnh sẽ thay đổi thành:
:: (venv) C:\path\to\semantic_eval>
```

### Ubuntu/Linux
```bash
# Di chuyển vào thư mục dự án
cd /path/to/semantic_eval

# Tạo môi trường ảo
python3.10 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate

# Dấu nhắc lệnh thay đổi thành:
# (venv) user@machine:/path/to/semantic_eval$
```

> **Kiểm tra môi trường đang hoạt động:**
> ```
> python --version   # phải ra Python 3.10.11
> which python       # (Linux) phải trỏ vào venv/bin/python
> ```

---

## BƯỚC 3 — CÀI ĐẶT THƯ VIỆN

Đảm bảo môi trường ảo đã được **kích hoạt** (thấy prefix `(venv)`) trước khi chạy lệnh này.

```bash
# Nâng cấp pip trước
pip install --upgrade pip

# Cài đặt tất cả thư viện từ file requirements.txt
pip install -r requirements.txt
```

### Chi tiết từng nhóm thư viện

#### 3.1 Xử lý dữ liệu
```bash
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 openpyxl==3.1.2
```

#### 3.2 PyTorch (chọn 1 trong 2 tùy cấu hình máy)
```bash
# Nếu KHÔNG có GPU (CPU only)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Nếu CÓ GPU NVIDIA (CUDA 11.8)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 3.3 Transformers & PhoBERT
```bash
pip install transformers==4.41.2 sentencepiece==0.2.0 tokenizers==0.19.1
```

#### 3.4 Tách từ tiếng Việt
```bash
pip install underthesea==6.8.4
```

#### 3.5 ELMo (chỉ cài nếu cần dùng mô hình ELMo)
```bash
# Lưu ý: allennlp yêu cầu thêm spacy
pip install allennlp==2.10.1 allennlp-models==2.10.1
```

---

## BƯỚC 4 — TẢI MÔ HÌNH PHO BERT (TỰ ĐỘNG)

Lần đầu chạy, chương trình sẽ **tự động tải** mô hình từ Hugging Face Hub:

| Mô hình            | Kích thước  | Tên trên HuggingFace             |
|--------------------|-------------|----------------------------------|
| PhoBERT-Base       | ~400 MB     | `vinai/phobert-base`             |
| PhoBERT-Large      | ~1.2 GB     | `vinai/phobert-large`            |
| BERT-Multilingual  | ~700 MB     | `bert-base-multilingual-cased`   |

Mô hình được cache tại:
- **Windows:** `C:\Users\<tên_user>\.cache\huggingface\hub\`
- **Linux:** `~/.cache/huggingface/hub/`

---

## BƯỚC 5 — CHẠY CHƯƠNG TRÌNH

```bash
# Đảm bảo môi trường ảo đang kích hoạt
# (venv) ...

# Tạo dữ liệu mẫu để kiểm thử (tùy chọn)
python create_sample_data.py

# Khởi động giao diện chính
python app.py
```

---

## HƯỚNG DẪN SỬ DỤNG GIAO DIỆN

```
┌─────────────────────────────────────────────────────────────────┐
│  Mô hình: [PhoBERT-Base ▼]  [📂 Tải file Excel]  [▶ Chạy]  ... │
├─────────────────────────────────────────────────────────────────┤
│  Hệ số tương quan:  Pearson: r = —    Spearman: ρ = —           │
├─────────────────────────────────────────────────────────────────┤
│  ID │ word1 │ pos1 │ word2 │ pos2 │ context1 │ context2 │ ... │ │
│  ── │ ───── │ ──── │ ───── │ ──── │ ──────── │ ──────── │ ... │ │
├─────────────────────────────────────────────────────────────────┤
│  Log: [Đang tải mô hình PhoBERT-Base ...]                       │
└─────────────────────────────────────────────────────────────────┘
```

### Các bước thao tác
1. **Chọn mô hình** từ dropdown (PhoBERT-Base / PhoBERT-Large / BERT-Multilingual / ELMo)
2. Nhấn **"📂 Tải file Excel"** → chọn file `.xlsx` chứa bộ dữ liệu
3. Nhấn **"▶ Chạy đánh giá"** → chờ quá trình xử lý (theo dõi log bên dưới)
4. Kết quả hiển thị tự động trong bảng, hệ số Pearson/Spearman xuất hiện ở thanh trên
5. Dùng **checkbox "Hiển thị cột s1–s10"** để ẩn/hiện các cột điểm thủ công
6. Nhấn **"💾 Xuất kết quả"** để lưu file `.xlsx` hoặc `.csv`

---

## XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi: `No module named 'tkinter'` (Ubuntu)
```bash
sudo apt install python3.10-tk -y
```

### Lỗi: `ModuleNotFoundError: No module named 'underthesea'`
```bash
pip install underthesea==6.8.4
```

### Lỗi: Kết nối HuggingFace thất bại (mạng chậm hoặc bị chặn)
```bash
# Đặt mirror (nếu ở mạng bị chặn HuggingFace)
set HF_ENDPOINT=https://hf-mirror.com   # Windows
export HF_ENDPOINT=https://hf-mirror.com  # Linux

# Hoặc tải thủ công và đặt đường dẫn local vào code
```

### Lỗi: `torch` không nhận GPU
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu ra False, cài lại torch với CUDA đúng phiên bản
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### Tắt môi trường ảo khi xong việc
```bash
deactivate
```

---

## CẤU TRÚC THƯ MỤC

```
semantic_eval/
├── app.py                  # Chương trình chính (giao diện + đánh giá)
├── requirements.txt        # Danh sách thư viện
├── create_sample_data.py   # Script tạo dữ liệu mẫu
├── README.md               # File hướng dẫn này
├── venv/                   # Môi trường ảo (tự tạo, không commit git)
└── sample_data.xlsx        # Dữ liệu mẫu (tạo bởi create_sample_data.py)
```
