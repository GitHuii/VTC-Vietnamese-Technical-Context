"""
core/models.py
Định nghĩa các lớp mô hình học sâu dùng để trích xuất vector ngữ cảnh.
Cập nhật: Hỗ trợ tự động chạy GPU (CUDA) và tích hợp XLM-RoBERTa.
"""
import torch
import numpy as np
from core.segmenter import segment_text, get_segmenter

_model_cache: dict = {}

class BaseModel:
    def load(self, log_fn=print):
        raise NotImplementedError
    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        raise NotImplementedError

def _find_target_span(all_tokens: list, target_tokens: list):
    # (Giữ nguyên như cũ)
    k = len(target_tokens)
    if k == 0: return None
    best_start, best_score = None, 0.0
    for i in range(len(all_tokens) - k + 1):
        score = sum(1 for a, b in zip(all_tokens[i:i+k], target_tokens) if a.lower() == b.lower()) / k
        if score > best_score:
            best_score, best_start = score, i
    return (best_start, k) if best_score >= 0.6 else None

# ══════════════════════════════════════════════
#  PHOBERT-BASE
# ══════════════════════════════════════════════
class PhoBERTModel(BaseModel):
    def __init__(self, variant: str = "vinai/phobert-base"):
        self.variant   = variant
        self.tokenizer = None
        self.model     = None
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, log_fn=print):
        key = self.variant
        if key in _model_cache:
            self.tokenizer, self.model = _model_cache[key]
            log_fn(f"[PhoBERT] Dùng lại model trên {self.device}: {key}")
        else:
            log_fn(f"[PhoBERT] Đang tải {key} lên {self.device} ...")
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(key)
                self.model = AutoModel.from_pretrained(key).to(self.device)
                self.model.eval()
                _model_cache[key] = (self.tokenizer, self.model)
                log_fn(f"[PhoBERT] Tải xong trên {self.device}: {key}")
            except Exception as e:
                raise RuntimeError(f"Không thể tải PhoBERT ({key}): {e}")

        get_segmenter(log_fn=log_fn)

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        segmented = segment_text(sentence)
        inputs = self.tokenizer(
            segmented, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        ).to(self.device) # <-- Đẩy input lên GPU
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[0]

        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            target_segmented = segment_text(target_word.lower())
            target_tokens = self.tokenizer.tokenize(target_segmented)
            
            span = _find_target_span(all_tokens, target_tokens)
            if span is not None:
                start, k = span
                # <-- Kéo về CPU trước khi sang Numpy
                return hidden[start: start + k].mean(dim=0).cpu().numpy() 

        return hidden[1:-1].mean(dim=0).cpu().numpy()

# ══════════════════════════════════════════════
#  BERT MULTILINGUAL
# ══════════════════════════════════════════════
class BERTMultilingualModel(BaseModel):
    VARIANT = "bert-base-multilingual-cased"

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, log_fn=print):
        key = self.VARIANT
        if key in _model_cache:
            self.tokenizer, self.model = _model_cache[key]
            log_fn(f"[mBERT] Dùng lại model trên {self.device}")
            return

        log_fn(f"[mBERT] Đang tải bert-base-multilingual-cased lên {self.device}...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            self.model     = AutoModel.from_pretrained(key).to(self.device)
            self.model.eval()
            _model_cache[key] = (self.tokenizer, self.model)
            log_fn("[mBERT] Tải xong.")
        except Exception as e:
            raise RuntimeError(f"Không thể tải mBERT: {e}")

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        inputs = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]

        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            target_tokens = self.tokenizer.tokenize(target_word)
            span = _find_target_span(all_tokens, target_tokens)
            if span is not None:
                start, k = span
                return hidden[start: start + k].mean(dim=0).cpu().numpy()

        return hidden[1:-1].mean(dim=0).cpu().numpy()

# ══════════════════════════════════════════════
#  XLM-ROBERTA
# ══════════════════════════════════════════════
class XLMRoBERTaModel(BaseModel):
    def __init__(self, variant: str = "xlm-roberta-base"):
        self.variant   = variant
        self.tokenizer = None
        self.model     = None
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, log_fn=print):
        key = self.variant
        if key in _model_cache:
            self.tokenizer, self.model = _model_cache[key]
            log_fn(f"[XLM-R] Dùng lại model trên {self.device}: {key}")
            return

        log_fn(f"[XLM-R] Đang tải {key} lên {self.device}...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            # Xóa output_hidden_states=True vì chỉ cần lấy lớp cuối
            self.model = AutoModel.from_pretrained(key).to(self.device)
            self.model.eval()
            _model_cache[key] = (self.tokenizer, self.model)
            log_fn(f"[XLM-R] Tải xong: {key}")
        except Exception as e:
            raise RuntimeError(f"Không thể tải XLM-RoBERTa ({key}): {e}")

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        inputs = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Chỉ lấy lớp ẩn cuối cùng như bình thường
        hidden = outputs.last_hidden_state[0]

        if target_word:
            encoding = self.tokenizer(
                sentence, return_tensors="pt",
                truncation=True, max_length=256,
                padding=True, return_offsets_mapping=True
            )
            offsets = encoding["offset_mapping"][0].tolist() 
            
            target_lower   = target_word.lower()
            sentence_lower = sentence.lower()
            char_start = sentence_lower.find(target_lower)
            
            if char_start != -1:
                char_end = char_start + len(target_lower)
                token_indices = [
                    i for i, (s, e) in enumerate(offsets)
                    if s < char_end and e > char_start and e > 0
                ]
                
                if token_indices:
                    return hidden[token_indices].mean(dim=0).cpu().numpy()

        return hidden[1:-1].mean(dim=0).cpu().numpy()

# ══════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════
MODEL_REGISTRY = {
    "PhoBERT-Base"      : lambda: PhoBERTModel("vinai/phobert-base"),
    "BERT-Multilingual" : BERTMultilingualModel,
    "XLMRoBERTa-Base"   : lambda: XLMRoBERTaModel("xlm-roberta-base"),
}

def create_model(model_name: str) -> BaseModel:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Mô hình '{model_name}' không hợp lệ. Chọn một trong: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()