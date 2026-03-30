"""
core/models.py
Định nghĩa các lớp mô hình học sâu dùng để trích xuất vector ngữ cảnh.

Tối ưu tốc độ so với phiên bản gốc:
  - Cache model toàn cục: mỗi variant chỉ tải 1 lần dù gọi nhiều lần
  - target_word tokenize được xử lý chuẩn xác để đảm bảo không sai lệch độ chính xác.
"""

import numpy as np
from core.segmenter import segment_text, get_segmenter

# ── Cache toàn cục: { variant_key: (tokenizer, model) } ──────
_model_cache: dict = {}


# ══════════════════════════════════════════════
#  LỚP CƠ SỞ
# ══════════════════════════════════════════════

class BaseModel:
    """Interface chung cho tất cả các mô hình."""

    def load(self, log_fn=print):
        raise NotImplementedError

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        raise NotImplementedError


# ══════════════════════════════════════════════
#  HÀM TÌM TOKEN MỤC TIÊU
# ══════════════════════════════════════════════

def _find_target_span(all_tokens: list, target_tokens: list):
    """
    Tìm vị trí bắt đầu của target_tokens trong all_tokens bằng
    cửa sổ trượt với ngưỡng overlap 60%.
    Trả về (start, k) hoặc None nếu không tìm thấy.
    """
    k = len(target_tokens)
    if k == 0:
        return None
    best_start, best_score = None, 0.0
    for i in range(len(all_tokens) - k + 1):
        score = sum(
            1 for a, b in zip(all_tokens[i:i+k], target_tokens)
            if a.lower() == b.lower()
        ) / k
        if score > best_score:
            best_score, best_start = score, i
    return (best_start, k) if best_score >= 0.6 else None


# ══════════════════════════════════════════════
#  PHOBERT (BASE & LARGE)
# ══════════════════════════════════════════════

class PhoBERTModel(BaseModel):
    """
    Mô hình PhoBERT — vinai/phobert-base hoặc vinai/phobert-large.
    """

    def __init__(self, variant: str = "vinai/phobert-base"):
        self.variant   = variant
        self.tokenizer = None
        self.model     = None

    def load(self, log_fn=print):
        key = self.variant
        if key in _model_cache:
            self.tokenizer, self.model = _model_cache[key]
            log_fn(f"[PhoBERT] Dùng lại model đã tải: {key}")
        else:
            log_fn(f"[PhoBERT] Đang tải {key} ...")
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(key)
                self.model     = AutoModel.from_pretrained(key)
                self.model.eval()
                _model_cache[key] = (self.tokenizer, self.model)
                log_fn(f"[PhoBERT] Tải xong: {key}")
            except Exception as e:
                raise RuntimeError(f"Không thể tải PhoBERT ({key}): {e}")

        # Khởi tạo singleton segmenter và log ngay khi load model
        get_segmenter(log_fn=log_fn)

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        import torch

        # Bước 1: tách từ câu ngữ cảnh
        segmented = segment_text(sentence)

        # Bước 2: tokenize và lấy hidden states
        inputs = self.tokenizer(
            segmented, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

        # Bước 3: tìm span của từ mục tiêu
        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # BẮT BUỘC gọi segment_text để tạo dấu '_' cho từ mục tiêu
            target_segmented = segment_text(target_word.lower())
            target_tokens = self.tokenizer.tokenize(target_segmented)
            
            span = _find_target_span(all_tokens, target_tokens)
            if span is not None:
                start, k = span
                # v(w, C) = (1/k) × Σ h_i  (công thức 2.5.3)
                return hidden[start: start + k].mean(dim=0).numpy()

        # Fallback: mean pooling bỏ [CLS] và [SEP]
        return hidden[1:-1].mean(dim=0).numpy()


# ══════════════════════════════════════════════
#  BERT MULTILINGUAL
# ══════════════════════════════════════════════

class BERTMultilingualModel(BaseModel):
    """
    BERT Multilingual Cased — bert-base-multilingual-cased.
    Không cần tách từ tiếng Việt (dùng WordPiece trực tiếp).
    """

    VARIANT = "bert-base-multilingual-cased"

    def __init__(self):
        self.tokenizer = None
        self.model     = None

    def load(self, log_fn=print):
        key = self.VARIANT
        if key in _model_cache:
            self.tokenizer, self.model = _model_cache[key]
            log_fn("[mBERT] Dùng lại model đã tải.")
            return

        log_fn("[mBERT] Đang tải bert-base-multilingual-cased ...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            self.model     = AutoModel.from_pretrained(key)
            self.model.eval()
            _model_cache[key] = (self.tokenizer, self.model)
            log_fn("[mBERT] Tải xong.")
        except Exception as e:
            raise RuntimeError(f"Không thể tải mBERT: {e}")

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]

        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # BỎ .lower() để WordPiece Cased hoạt động chính xác
            target_tokens = self.tokenizer.tokenize(target_word)
            
            span = _find_target_span(all_tokens, target_tokens)
            if span is not None:
                start, k = span
                return hidden[start: start + k].mean(dim=0).numpy()

        return hidden[1:-1].mean(dim=0).numpy()


# ══════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════

MODEL_REGISTRY = {
    "PhoBERT-Base"      : lambda: PhoBERTModel("vinai/phobert-base"),
    "PhoBERT-Large"     : lambda: PhoBERTModel("vinai/phobert-large"),
    "BERT-Multilingual" : BERTMultilingualModel,
}


def create_model(model_name: str) -> BaseModel:
    """Tạo instance mô hình theo tên. Raise ValueError nếu không hợp lệ."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Mô hình '{model_name}' không hợp lệ. "
            f"Chọn một trong: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()