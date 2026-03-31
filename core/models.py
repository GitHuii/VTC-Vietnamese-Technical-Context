"""
core/models.py
Định nghĩa các lớp mô hình học sâu dùng để trích xuất vector ngữ cảnh.
Cập nhật: Chỉ áp dụng trung bình 4 lớp cuối cho PhoBERT-Large để sửa lỗi phân bố điểm.
PhoBERT-Base và mBERT vẫn giữ nguyên cách lấy lớp cuối cùng.
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
                
                # CHỈ SỬA LARGE: Kích hoạt output_hidden_states cho bản Large
                is_large = "large" in key.lower()
                self.model = AutoModel.from_pretrained(
                    key, 
                    output_hidden_states=is_large
                )
                
                self.model.eval()
                _model_cache[key] = (self.tokenizer, self.model)
                log_fn(f"[PhoBERT] Tải xong: {key}")
            except Exception as e:
                raise RuntimeError(f"Không thể tải PhoBERT ({key}): {e}")

        # Khởi tạo singleton segmenter
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
            
            # ─── PHÂN TÁCH LOGIC CHO BASE VÀ LARGE ───
            if "large" in self.variant.lower():
                # CẢI THIỆN CHO LARGE: Lấy trung bình 4 lớp cuối
                all_layers = outputs.hidden_states # List của 25 tensors
                last_4_layers = torch.stack(all_layers[-4:]) # Lấy 4 lớp cuối
                # Tính trung bình các lớp, sau đó lấy câu đầu tiên [0]
                hidden = last_4_layers.mean(dim=0)[0] 
            else:
                # GIỮ NGUYÊN CHO BASE: Lấy duy nhất lớp cuối cùng
                hidden = outputs.last_hidden_state[0]

        # Bước 3: tìm span của từ mục tiêu
        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            target_segmented = segment_text(target_word.lower())
            target_tokens = self.tokenizer.tokenize(target_segmented)
            
            span = _find_target_span(all_tokens, target_tokens)
            if span is not None:
                start, k = span
                # v(w, C) = (1/k) × Σ h_i
                return hidden[start: start + k].mean(dim=0).numpy()

        # Fallback: mean pooling bỏ [CLS] và [SEP]
        return hidden[1:-1].mean(dim=0).numpy()


# ══════════════════════════════════════════════
#  BERT MULTILINGUAL
# ══════════════════════════════════════════════

class BERTMultilingualModel(BaseModel):
    """
    BERT Multilingual Cased — bert-base-multilingual-cased.
    GIỮ NGUYÊN: Lấy lớp cuối cùng (last_hidden_state).
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
            # mBERT giữ nguyên, không cần output_hidden_states
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
        
        # Mặc định lấy lớp cuối cùng
        hidden = outputs.last_hidden_state[0]

        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
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
    """Tạo instance mô hình theo tên."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Mô hình '{model_name}' không hợp lệ. "
            f"Chọn một trong: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()