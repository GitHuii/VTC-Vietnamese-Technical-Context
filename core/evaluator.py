"""
core/evaluator.py
Pipeline đánh giá độ tương đồng ngữ nghĩa cho toàn bộ bộ dữ liệu.
"""
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

from scipy.stats import pearsonr, spearmanr
from typing import Callable

from core.utils  import extract_target_word, normalize_text, cosine_similarity, scale_cos_to_10
from core.models import BaseModel


def evaluate_dataset(
    df: pd.DataFrame,
    model: BaseModel,
    log_fn: Callable[[str], None] = print,
) -> pd.DataFrame:
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
            
            # Sử dụng trực tiếp hàm quy đổi, không qua Min-Max Scaling
            model_scores.append(scale_cos_to_10(cos))
        except Exception as e:
            log_fn(f"  [!] Lỗi tại mẫu {idx}: {e}")
            raw_cosines.append(None)
            model_scores.append(None)

        cur = len(raw_cosines)
        if cur % 50 == 0 or cur == total:
            log_fn(f"  Tiến độ: {cur}/{total}")

    result = df.copy()
    result["raw_cosine"]  = raw_cosines
    result["model_score"] = model_scores

    return result

def compute_correlations(df: pd.DataFrame) -> dict | None:
    # 1. Kiểm tra cột dữ liệu
    if "avg_score" not in df.columns or "model_score" not in df.columns:
        return None
        
    # 2. Lọc dữ liệu hợp lệ (bỏ NaN)
    valid = df[["avg_score", "model_score"]].dropna()
    if len(valid) < 2:
        return None
        
    y_true = valid["avg_score"].values
    y_pred = valid["model_score"].values

    # 3. Tính toán Hệ số tương quan & p-value
    p_r, p_p = pearsonr(y_true, y_pred)
    s_r, s_p = spearmanr(y_true, y_pred)
    k_t, k_p = kendalltau(y_true, y_pred)

    # 4. Tính toán Sai số (MAE, RMSE)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # 5. Tính Hệ số xác định (R²)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Trả về toàn bộ kết quả đã làm tròn 4 chữ số thập phân
    return {
        "pearson_r"  : round(float(p_r), 4),
        "pearson_p"  : round(float(p_p), 4),
        "spearman_r" : round(float(s_r), 4),
        "spearman_p" : round(float(s_p), 4),
        "kendall_tau": round(float(k_t), 4),
        "kendall_p"  : round(float(k_p), 4),
        "mae"        : round(float(mae), 4),
        "rmse"       : round(float(rmse), 4),
        "r2"         : round(float(r2), 4),
        "n_valid"    : len(valid),
    }