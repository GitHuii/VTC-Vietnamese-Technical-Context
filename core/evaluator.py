"""
core/evaluator.py
Pipeline đánh giá độ tương đồng ngữ nghĩa cho toàn bộ bộ dữ liệu.

Quy trình (mục 2.5 tài liệu nghiên cứu):
  1. Chuẩn hóa văn bản → tách từ RDRSegmenter → trích xuất vector từ mục tiêu
  2. Tính Cosine Similarity
  3. Quy đổi: score_model = cos(u, v) × 10  (mục 2.5.5)
  4. Tính Pearson / Spearman (avg_score, model_score)
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Callable

from core.utils  import extract_target_word, normalize_text, cosine_similarity, scale_cos_to_10
from core.models import BaseModel


def evaluate_dataset(
    df: pd.DataFrame,
    model: BaseModel,
    log_fn: Callable[[str], None] = print,
) -> pd.DataFrame:
    """
    Duyệt toàn bộ bộ dữ liệu, tính model_score cho từng mẫu.

    Tham số:
        df     : DataFrame với các cột context1, context2, avg_score, ...
        model  : instance mô hình đã được load()
        log_fn : hàm ghi log (print hoặc callback UI)

    Trả về:
        DataFrame gốc đã thêm cột raw_cosine và model_score.
    """
    raw_cosines  = []
    model_scores = []
    total = len(df)

    for idx, row in df.iterrows():
        ctx1 = str(row.get("context1", ""))
        ctx2 = str(row.get("context2", ""))

        target1, sent1 = extract_target_word(ctx1)
        target2, sent2 = extract_target_word(ctx2)

        sent1 = normalize_text(sent1)
        sent2 = normalize_text(sent2)

        try:
            vec1 = model.get_vector(sent1, target1 or "")
            vec2 = model.get_vector(sent2, target2 or "")
            cos  = cosine_similarity(vec1, vec2)
            raw_cosines.append(cos)
            model_scores.append(scale_cos_to_10(cos))
        except Exception as e:
            log_fn(f"  [!] Lỗi tại mẫu {idx}: {e}")
            raw_cosines.append(None)
            model_scores.append(None)

        cur = len(raw_cosines)
        if cur % 50 == 0 or cur == total:
            log_fn(f"  Tiến độ: {cur}/{total}")

    result = df.copy()
    result["raw_cosine"]  = raw_cosines   # cosine thô để debug/phân tích
    result["model_score"] = model_scores
    valid = [(i, s) for i, s in enumerate(model_scores) if s is not None]
    if valid:
        indices, scores = zip(*valid)
        s_min = min(scores)
        s_max = max(scores)
        if s_max > s_min:  # tránh chia 0
            scaled = [None] * len(model_scores)
            for i, s in valid:
                scaled[i] = round((s - s_min) / (s_max - s_min) * 10, 4)
            result["model_score"] = scaled

    return result


def compute_correlations(df: pd.DataFrame) -> dict | None:
    """
    Tính Pearson và Spearman giữa avg_score (con người) và model_score.
    Trả về dict với các key: pearson_r, pearson_p, spearman_r, spearman_p, n_valid.
    Trả về None nếu không đủ dữ liệu hợp lệ.
    """
    if "avg_score" not in df.columns or "model_score" not in df.columns:
        return None
    valid = df[["avg_score", "model_score"]].dropna()
    if len(valid) < 2:
        return None
    p_r, p_p = pearsonr(valid["avg_score"],  valid["model_score"])
    s_r, s_p = spearmanr(valid["avg_score"], valid["model_score"])
    return {
        "pearson_r" : round(float(p_r), 4),
        "pearson_p" : round(float(p_p), 4),
        "spearman_r": round(float(s_r), 4),
        "spearman_p": round(float(s_p), 4),
        "n_valid"   : len(valid),
    }
