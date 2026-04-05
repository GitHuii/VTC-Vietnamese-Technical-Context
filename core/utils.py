"""
core/utils.py
Các hàm tiện ích dùng chung: trích xuất từ mục tiêu, chuẩn hóa,
tính Cosine Similarity và quy đổi điểm.
"""

import re
import math
import numpy as np


def extract_target_word(context: str):
    match = re.search(r'<b>(.*?)</b>', context, re.IGNORECASE)
    if not match:
        return None, context
    target = match.group(1).strip()
    clean  = re.sub(r'<b>(.*?)</b>', r'\1', context, flags=re.IGNORECASE).strip()
    return target, clean


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def scale_cos_to_10(cos_sim: float) -> float:
    """
    Quy đổi Cosine Similarity [-1, 1] về thang 0–10.
    Mặc định sử dụng công thức biến đổi tuyến tính để bảo toàn hệ số tương quan.
    """
    if cos_sim is None:
        return 0.0

    cos_val = float(cos_sim)

    scaled_val = max(0.0, cos_val) * 10.0

    scaled_val = max(0.0, min(10.0, scaled_val))
    
    return round(scaled_val, 4)