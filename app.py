"""
Hệ thống đánh giá độ tương đồng ngữ nghĩa thuật ngữ kỹ thuật
Khóa luận tốt nghiệp - Nhóm: Nguyễn Viết Huy & Hoàng Thanh Chiến
ĐHKT-KT Công Nghiệp - Khoa CNTT
"""

import sys
import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ─────────────────────────────────────────────
# Biến toàn cục quản lý mô hình đã tải
# ─────────────────────────────────────────────
_loaded_models = {}


# ══════════════════════════════════════════════
#  CÁC HÀM TIỆN ÍCH
# ══════════════════════════════════════════════

def extract_target_word(context: str):
    """
    Trích xuất từ mục tiêu được bọc trong thẻ <b>...</b>.
    Trả về (từ_mục_tiêu, câu_đã_bỏ_thẻ).
    """
    match = re.search(r'<b>(.*?)</b>', context, re.IGNORECASE)
    if not match:
        return None, context
    target = match.group(1).strip()
    clean = re.sub(r'<b>(.*?)</b>', r'\1', context, flags=re.IGNORECASE).strip()
    return target, clean


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt cơ bản."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Tính Cosine Similarity giữa hai vector."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def scale_cos_to_10(cos_sim: float) -> float:
    """
    Quy đổi Cosine Similarity về thang 0–10 theo công thức tuyến tính:
        score_model = cos(u, v) × 10
    Cosine Similarity của vector BERT/PhoBERT thực tế nằm trong [0, 1]
    do các giá trị hidden state không âm sau hàm kích hoạt, nên nhân
    thẳng với 10 là đủ để đưa về thang điểm tương đương avg_score con người.
    """
    return round(float(cos_sim) * 10, 4)


# ══════════════════════════════════════════════
#  LỚP MÔ HÌNH
# ══════════════════════════════════════════════

class BaseModel:
    """Lớp cơ sở cho tất cả các mô hình."""

    def load(self, log_fn=print):
        raise NotImplementedError

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        raise NotImplementedError


# ─────────────────── PhoBERT ───────────────────
class PhoBERTModel(BaseModel):
    def __init__(self, variant="vinai/phobert-base"):
        self.variant = variant
        self.tokenizer = None
        self.model = None

    def load(self, log_fn=print):
        key = self.variant
        if key in _loaded_models:
            self.tokenizer, self.model = _loaded_models[key]
            log_fn(f"[PhoBERT] Sử dụng lại mô hình đã tải: {key}")
            return

        log_fn(f"[PhoBERT] Đang tải mô hình {key} ...")
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            self.model = AutoModel.from_pretrained(key)
            self.model.eval()
            _loaded_models[key] = (self.tokenizer, self.model)
            log_fn(f"[PhoBERT] Tải xong: {key}")
            # Kiểm tra VnCoreNLP jar ngay khi load để log trạng thái tách từ
            jar_name = "VnCoreNLP-1.1.1.jar"
            app_dir  = os.path.dirname(os.path.abspath(__file__))
            cwd      = os.getcwd()
            jar_candidates = [
                os.path.join(cwd,     jar_name),
                os.path.join(cwd,     "vncorenlp", jar_name),
                os.path.join(app_dir, jar_name),
                os.path.join(app_dir, "vncorenlp", jar_name),
            ]
            jar_found = next((p for p in jar_candidates if os.path.isfile(p)), None)
            try:
                import vncorenlp as _vnc  # noqa: kiểm tra có cài không
                if jar_found:
                    log_fn(f"[PhoBERT] Tách từ: RDRSegmenter  ← {jar_found}")
                else:
                    log_fn(f"[PhoBERT] Cảnh báo: Không thấy {jar_name} → fallback underthesea")
                    log_fn(f"[PhoBERT] Đã tìm ở: {chr(10).join(jar_candidates)}")
            except ImportError:
                log_fn("[PhoBERT] Cảnh báo: vncorenlp chưa cài → fallback underthesea")
        except Exception as e:
            raise RuntimeError(f"Không thể tải PhoBERT ({key}): {e}")

    def _segment(self, text: str, log_fn=print) -> str:
        """
        Tách từ tiếng Việt bằng RDRSegmenter thuộc bộ VnCoreNLP.
        RDRSegmenter là công cụ tách từ chính thức được dùng khi huấn luyện
        PhoBERT, đảm bảo dữ liệu đầu vào nhất quán với quá trình pre-training.
        Các từ ghép được nối bằng dấu gạch dưới: "thuật_toán", "học_sâu".
        Fallback về underthesea nếu VnCoreNLP chưa cài, và giữ nguyên
        văn bản nếu cả hai đều không khả dụng.
        """
        # Ưu tiên 1: RDRSegmenter (VnCoreNLP) — khuyến nghị cho PhoBERT
        try:
            from vncorenlp import VnCoreNLP
            if not hasattr(self, '_rdrsegmenter'):
                # Tìm file jar theo thứ tự ưu tiên:
                # - thư mục làm việc hiện tại (cwd)
                # - thư mục vncorenlp/ trong cwd  ← cấu trúc thực tế của dự án
                # - cùng thư mục với app.py
                # - vncorenlp/ cạnh app.py
                app_dir = os.path.dirname(os.path.abspath(__file__))
                cwd     = os.getcwd()
                jar_name = "VnCoreNLP-1.1.1.jar"
                jar_candidates = [
                    os.path.join(cwd,     jar_name),
                    os.path.join(cwd,     "vncorenlp", jar_name),
                    os.path.join(app_dir, jar_name),
                    os.path.join(app_dir, "vncorenlp", jar_name),
                ]
                jar_path = next((p for p in jar_candidates if os.path.isfile(p)), None)
                log_fn(f"Jar path: {jar_path}")
                if jar_path is None:
                    # Log rõ để dễ debug thay vì im lặng
                    searched = "\n  ".join(jar_candidates)
                    raise FileNotFoundError(
                        f"Không tìm thấy {jar_name}.\n"
                        f"Đã tìm ở:\n  {searched}\n"
                        "→ Chuyển sang fallback underthesea."
                    )
                self._rdrsegmenter = VnCoreNLP(
                    jar_path, annotators="wseg", max_heap_size="-Xmx500m"
                )
            sentences = self._rdrsegmenter.tokenize(text)
            # sentences là list of list of tokens → nối thành chuỗi
            return " ".join(" ".join(sent) for sent in sentences)
        except FileNotFoundError as e:
            # In ra để log nhưng vẫn tiếp tục bằng fallback
            print(f"[_segment] {e}")
        except Exception as e:
            print(f"[_segment] VnCoreNLP lỗi: {e} — chuyển sang fallback.")

        # Ưu tiên 2: underthesea (fallback)
        try:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
        except Exception as e:
            print(f"[_segment] underthesea lỗi: {e} — giữ nguyên văn bản.")

        # Ưu tiên 3: giữ nguyên văn bản
        return text

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        """
        Trích xuất vector ngữ cảnh hóa của từ mục tiêu theo đúng quy trình:
        1. Tách từ tiếng Việt bằng RDRSegmenter (VnCoreNLP)
        2. Tokenize bằng PhoBERT BPE (64k subword)
        3. Tìm vị trí các subword token của từ mục tiêu trong chuỗi token
        4. Lấy trung bình vector lớp ẩn cuối của các subword đó:
               v(w, C) = (1/k) × Σ h_i  (công thức mục 2.5.3)
        Fallback về mean pooling toàn câu nếu không tìm được từ mục tiêu.
        """
        import torch
        # Bước 1: tách từ tiếng Việt
        segmented = self._segment(sentence)

        # Bước 2: tokenize toàn câu để lấy hidden states
        inputs = self.tokenizer(
            segmented,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # last_hidden_state: [1, seq_len, hidden_size]
        hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

        # Bước 3: tìm vị trí subword token của từ mục tiêu
        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            )
            # Tách từ mục tiêu thành subword tokens để so sánh
            target_segmented = self._segment(target_word.lower())
            target_tokens = self.tokenizer.tokenize(target_segmented)

            best_start = None
            best_overlap = 0.0

            if target_tokens:
                k = len(target_tokens)
                for i in range(len(all_tokens) - k + 1):
                    window = all_tokens[i: i + k]
                    overlap = sum(
                        1 for a, b in zip(window, target_tokens)
                        if a.lower() == b.lower()
                    )
                    ratio = overlap / k
                    if ratio > best_overlap:
                        best_overlap = ratio
                        best_start = i

            # Ngưỡng overlap 60% — giống code mẫu (Gradio)
            if best_start is not None and best_overlap >= 0.6:
                k = len(target_tokens)
                # v(w, C) = (1/k) × Σ h_i  (công thức 2.5.3 trong tài liệu)
                vec = hidden[best_start: best_start + k].mean(dim=0).numpy()
                return vec

        # Fallback: mean pooling toàn câu (bỏ [CLS] và [SEP])
        vec = hidden[1:-1].mean(dim=0).numpy()
        return vec


# ─────────────────── mBERT ───────────────────
class BERTMultilingualModel(BaseModel):
    def __init__(self):
        self.variant = "bert-base-multilingual-cased"
        self.tokenizer = None
        self.model = None

    def load(self, log_fn=print):
        key = self.variant
        if key in _loaded_models:
            self.tokenizer, self.model = _loaded_models[key]
            log_fn("[mBERT] Sử dụng lại mô hình đã tải.")
            return

        log_fn("[mBERT] Đang tải mô hình bert-base-multilingual-cased ...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            self.model = AutoModel.from_pretrained(key)
            self.model.eval()
            _loaded_models[key] = (self.tokenizer, self.model)
            log_fn("[mBERT] Tải xong.")
        except Exception as e:
            raise RuntimeError(f"Không thể tải mBERT: {e}")

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]  # [seq_len, hidden]

        if target_word:
            all_tokens = self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            )
            target_tokens = self.tokenizer.tokenize(target_word.lower())
            best_start, best_overlap = None, 0.0
            if target_tokens:
                k = len(target_tokens)
                for i in range(len(all_tokens) - k + 1):
                    window = all_tokens[i: i + k]
                    overlap = sum(
                        1 for a, b in zip(window, target_tokens)
                        if a.lower() == b.lower()
                    )
                    ratio = overlap / k
                    if ratio > best_overlap:
                        best_overlap = ratio
                        best_start = i
            if best_start is not None and best_overlap >= 0.6:
                k = len(target_tokens)
                return hidden[best_start: best_start + k].mean(dim=0).numpy()

        return hidden[1:-1].mean(dim=0).numpy()


# ─────────────────── ELMo ───────────────────
class ELMoModel(BaseModel):
    def __init__(self):
        self.elmo = None

    def load(self, log_fn=print):
        key = "elmo"
        if key in _loaded_models:
            self.elmo = _loaded_models[key]
            log_fn("[ELMo] Sử dụng lại mô hình đã tải.")
            return

        log_fn("[ELMo] Đang tải ELMo (allennlp) ...")
        try:
            from allennlp.modules.elmo import Elmo, batch_to_ids
            options_file = (
                "https://allennlp.s3.amazonaws.com/models/elmo/"
                "2x4096_512_2048cnn_2xhighway/"
                "elmo_2x4096_512_2048cnn_2xhighway_options.json"
            )
            weight_file = (
                "https://allennlp.s3.amazonaws.com/models/elmo/"
                "2x4096_512_2048cnn_2xhighway/"
                "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            )
            self.elmo = Elmo(options_file, weight_file, num_output_representations=1)
            self.elmo.eval()
            _loaded_models[key] = self.elmo
            log_fn("[ELMo] Tải xong.")
        except Exception as e:
            raise RuntimeError(
                f"Không thể tải ELMo: {e}\n"
                "Hãy chắc chắn đã cài: pip install allennlp allennlp-models"
            )

    def get_vector(self, sentence: str, target_word: str) -> np.ndarray:
        import torch
        from allennlp.modules.elmo import batch_to_ids
        tokens = sentence.split()
        char_ids = batch_to_ids([tokens])
        with torch.no_grad():
            embeddings = self.elmo(char_ids)
        # representations[0]: [1, seq_len, 1024]
        vec = embeddings["elmo_representations"][0].squeeze(0).mean(dim=0).numpy()
        return vec


# ══════════════════════════════════════════════
#  FACTORY TẠO MÔ HÌNH
# ══════════════════════════════════════════════

def create_model(model_name: str) -> BaseModel:
    mapping = {
        "PhoBERT-Base":        PhoBERTModel("vinai/phobert-base"),
        "PhoBERT-Large":       PhoBERTModel("vinai/phobert-large"),
        "BERT-Multilingual":   BERTMultilingualModel(),
        "ELMo":                ELMoModel(),
    }
    if model_name not in mapping:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")
    return mapping[model_name]


# ══════════════════════════════════════════════
#  HÀM ĐÁNH GIÁ CHÍNH
# ══════════════════════════════════════════════

def evaluate_dataset(df: pd.DataFrame, model: BaseModel, log_fn=print) -> pd.DataFrame:
    """
    Quy trình đánh giá theo tài liệu nghiên cứu mục 2.5:
    1. Với mỗi mẫu: chuẩn hóa → tách từ RDRSegmenter → trích xuất vector
       từ mục tiêu → tính Cosine Similarity
    2. Quy đổi: score_model = cos(u, v) × 10  (mục 2.5.5)
    3. Tính Pearson/Spearman(avg_score, model_score)
    """
    raw_cosines = []
    model_scores = []
    total = len(df)

    for idx, row in df.iterrows():
        ctx1 = str(row.get("context1", ""))
        ctx2 = str(row.get("context2", ""))

        target1, sentence1 = extract_target_word(ctx1)
        target2, sentence2 = extract_target_word(ctx2)

        sentence1 = normalize_text(sentence1)
        sentence2 = normalize_text(sentence2)

        try:
            vec1 = model.get_vector(sentence1, target1 or "")
            vec2 = model.get_vector(sentence2, target2 or "")
            cos = cosine_similarity(vec1, vec2)
            raw_cosines.append(cos)
            # score_model = cos(u, v) × 10  (tài liệu mục 2.5.5)
            model_scores.append(scale_cos_to_10(cos))
        except Exception as e:
            log_fn(f"  [!] Lỗi tại mẫu {idx}: {e}")
            raw_cosines.append(None)
            model_scores.append(None)

        cur = len(raw_cosines)
        if cur % 50 == 0 or cur == total:
            log_fn(f"  Tiến độ: {cur}/{total}")

    df = df.copy()
    df["raw_cosine"] = raw_cosines   # cosine thô để debug/phân tích
    df["model_score"] = model_scores
    return df


# ══════════════════════════════════════════════
#  GIAO DIỆN TKINTER
# ══════════════════════════════════════════════

SCORE_COLS = [f"s{i}" for i in range(1, 11)]
MODEL_CHOICES = ["PhoBERT-Base", "PhoBERT-Large", "BERT-Multilingual", "ELMo"]

ALL_COLUMNS = (
    ["ID", "word1", "pos1", "word2", "pos2", "context1", "context2"]
    + SCORE_COLS
    + ["avg_score", "model_score"]
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đánh giá Độ Tương Đồng Ngữ Nghĩa — Khóa Luận Tốt Nghiệp")
        self.geometry("1280x780")
        self.resizable(True, True)

        self.df_raw: pd.DataFrame = None
        self.df_result: pd.DataFrame = None
        self._show_scores = tk.BooleanVar(value=True)
        self._selected_model = tk.StringVar(value="PhoBERT-Base")

        self._build_ui()

    # ─── Xây dựng giao diện ───
    def _build_ui(self):
        # ── Thanh điều khiển trên cùng
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Mô hình:").pack(side="left", padx=4)
        model_cb = ttk.Combobox(
            ctrl, textvariable=self._selected_model,
            values=MODEL_CHOICES, state="readonly", width=20
        )
        model_cb.pack(side="left", padx=4)

        ttk.Button(ctrl, text="📂 Tải file Excel", command=self._load_excel).pack(side="left", padx=8)
        ttk.Button(ctrl, text="▶ Chạy đánh giá",  command=self._run_eval).pack(side="left", padx=4)
        ttk.Button(ctrl, text="💾 Xuất kết quả",  command=self._export).pack(side="left", padx=4)

        ttk.Checkbutton(
            ctrl, text="Hiển thị cột s1–s10",
            variable=self._show_scores,
            command=self._toggle_score_cols
        ).pack(side="left", padx=12)

        self._lbl_file = ttk.Label(ctrl, text="Chưa tải file", foreground="gray")
        self._lbl_file.pack(side="left", padx=8)

        # ── Nhãn thống kê tương quan
        stat_frame = ttk.LabelFrame(self, text="Hệ số tương quan", padding=6)
        stat_frame.pack(fill="x", padx=8, pady=2)

        self._lbl_pearson  = ttk.Label(stat_frame, text="Pearson:  —")
        self._lbl_spearman = ttk.Label(stat_frame, text="Spearman: —")
        self._lbl_pearson.pack(side="left", padx=16)
        self._lbl_spearman.pack(side="left", padx=16)

        # ── Bảng kết quả
        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._tree = ttk.Treeview(table_frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(table_frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # ── Log console
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="x", padx=8, pady=4)

        self._log_text = tk.Text(log_frame, height=6, state="disabled",
                                  wrap="word", font=("Consolas", 9))
        log_sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        self._log_text.pack(side="left", fill="x", expand=True)
        log_sb.pack(side="right", fill="y")

    # ─── Ghi log ───
    def _log(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.update_idletasks()

    # ─── Tải file Excel ───
    def _load_excel(self):
        path = filedialog.askopenfilename(
            title="Chọn file dữ liệu",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.df_raw = pd.read_excel(path)
            self._lbl_file.configure(text=os.path.basename(path), foreground="black")
            self._log(f"Đã tải: {path} — {len(self.df_raw)} mẫu")
            self._populate_table(self.df_raw)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không đọc được file:\n{e}")

    # ─── Chạy đánh giá ───
    def _run_eval(self):
        if self.df_raw is None:
            messagebox.showwarning("Chưa có dữ liệu", "Hãy tải file Excel trước.")
            return
        model_name = self._selected_model.get()
        self._log(f"\n=== Bắt đầu đánh giá với mô hình: {model_name} ===")

        def _worker():
            try:
                model = create_model(model_name)
                model.load(log_fn=self._log)
                df_out = evaluate_dataset(self.df_raw, model, log_fn=self._log)
                self.df_result = df_out
                self.after(0, lambda: self._on_eval_done(df_out))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
                self.after(0, lambda: self._log(f"[LỖI] {e}"))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _on_eval_done(self, df: pd.DataFrame):
        self._populate_table(df)
        self._compute_stats(df)
        self._log("=== Đánh giá hoàn tất ===\n")

    # ─── Điền dữ liệu vào bảng ───
    def _populate_table(self, df: pd.DataFrame):
        cols = [c for c in ALL_COLUMNS if c in df.columns]
        if not self._show_scores.get():
            cols = [c for c in cols if c not in SCORE_COLS]

        self._tree.delete(*self._tree.get_children())
        self._tree["columns"] = cols

        col_widths = {
            "ID": 45, "word1": 90, "pos1": 50, "word2": 90, "pos2": 50,
            "context1": 250, "context2": 250,
            "avg_score": 80, "model_score": 90,
            **{f"s{i}": 48 for i in range(1, 11)}
        }
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=col_widths.get(c, 70), anchor="center", stretch=False)

        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    v = f"{v:.4f}"
                vals.append(str(v) if v is not None else "")
            self._tree.insert("", "end", values=vals)

    # ─── Ẩn/hiện cột s1-s10 ───
    def _toggle_score_cols(self):
        df = self.df_result if self.df_result is not None else self.df_raw
        if df is not None:
            self._populate_table(df)

    # ─── Tính hệ số tương quan ───
    def _compute_stats(self, df: pd.DataFrame):
        if "avg_score" not in df.columns or "model_score" not in df.columns:
            return
        valid = df[["avg_score", "model_score"]].dropna()
        if len(valid) < 2:
            return
        p_r, p_p = pearsonr(valid["avg_score"], valid["model_score"])
        s_r, s_p = spearmanr(valid["avg_score"], valid["model_score"])
        self._lbl_pearson.configure(
            text=f"Pearson:  r = {p_r:.4f}  (p = {p_p:.4f})"
        )
        self._lbl_spearman.configure(
            text=f"Spearman: ρ = {s_r:.4f}  (p = {s_p:.4f})"
        )
        self._log(f"Pearson  r = {p_r:.4f} | Spearman ρ = {s_r:.4f}")

    # ─── Xuất kết quả ───
    def _export(self):
        if self.df_result is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy chạy đánh giá trước.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")]
        )
        if not path:
            return
        if path.endswith(".csv"):
            self.df_result.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            self.df_result.to_excel(path, index=False)
        self._log(f"Đã xuất kết quả: {path}")
        messagebox.showinfo("Thành công", f"Đã lưu tại:\n{path}")


# ══════════════════════════════════════════════
#  ĐIỂM VÀO CHƯƠNG TRÌNH
# ══════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()