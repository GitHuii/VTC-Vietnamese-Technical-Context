"""
core/utils.py
Các hàm tiện ích dùng chung: trích xuất từ mục tiêu, chuẩn hóa,
tính Cosine Similarity và quy đổi điểm.
"""

import re
import numpy as np


def extract_target_word(context: str):
    """
    Trích xuất từ mục tiêu được bọc trong thẻ <b>...</b>.
    Trả về (từ_mục_tiêu, câu_đã_bỏ_thẻ).
    Ví dụ:
        "Hệ thống <b>mạng</b> nội bộ..." → ("mạng", "Hệ thống mạng nội bộ...")
    """
    match = re.search(r'<b>(.*?)</b>', context, re.IGNORECASE)
    if not match:
        return None, context
    target = match.group(1).strip()
    clean  = re.sub(r'<b>(.*?)</b>', r'\1', context, flags=re.IGNORECASE).strip()
    return target, clean


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt cơ bản:
      - Loại bỏ khoảng trắng thừa
    Không chuyển thường để giữ nguyên tính nhất quán với
    dữ liệu huấn luyện của PhoBERT.
    """
    return re.sub(r'\s+', ' ', text.strip())


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Tính Cosine Similarity giữa hai vector đặc trưng.
    Công thức: cos(u, v) = (u · v) / (||u|| × ||v||)
    Trả về 0.0 nếu một trong hai vector là vector không.
    """
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def scale_cos_to_10(cos_sim: float) -> float:
    """
    Quy đổi Cosine Similarity về thang 0–10 theo công thức tuyến tính:
        score_model = cos(u, v) × 10
    Vector BERT/PhoBERT thực tế nằm trong [0, 1] do hidden state
    không âm sau hàm kích hoạt (mục 2.5.5 tài liệu nghiên cứu).
    """
    return round(float(cos_sim) * 10, 4)
