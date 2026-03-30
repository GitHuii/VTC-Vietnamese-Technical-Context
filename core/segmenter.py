"""
core/segmenter.py
Singleton RDRSegmenter — khởi tạo VnCoreNLP đúng 1 lần duy nhất
cho toàn bộ phiên làm việc, tránh overhead khởi động JVM lặp lại.

Thứ tự ưu tiên:
  1. RDRSegmenter (VnCoreNLP) — công cụ chính thức dùng khi pre-train PhoBERT
  2. underthesea              — fallback nếu VnCoreNLP không khả dụng
  3. Giữ nguyên văn bản       — fallback cuối cùng
"""

import os

# ── Singleton instance — chỉ khởi tạo 1 lần ──────────────────
_segmenter_instance = None
_segmenter_ready    = False   # True = đã thử khởi tạo (dù thành công hay thất bại)
_underthesea_tokenize = None  # Cache hàm fallback để tránh import lại nhiều lần

def _find_jar() -> str | None:
    """
    Tìm file VnCoreNLP-1.1.1.jar theo các vị trí phổ biến.
    Ưu tiên: cwd → cwd/vncorenlp/ → app_dir → app_dir/vncorenlp/
    """
    jar_name = "VnCoreNLP-1.1.1.jar"
    app_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd      = os.getcwd()
    candidates = [
        os.path.join(cwd,     jar_name),
        os.path.join(cwd,     "vncorenlp", jar_name),
        os.path.join(app_dir, jar_name),
        os.path.join(app_dir, "vncorenlp", jar_name),
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def get_segmenter(log_fn=print):
    """
    Trả về singleton RDRSegmenter.
    Lần đầu gọi: khởi tạo VnCoreNLP và log kết quả.
    Các lần sau: trả về instance đã tạo ngay lập tức (O(1)).
    """
    global _segmenter_instance, _segmenter_ready

    if _segmenter_ready:
        return _segmenter_instance   # đã khởi tạo → trả về ngay

    _segmenter_ready = True          # đánh dấu đã thử, tránh retry vô hạn

    jar = _find_jar()
    if jar is None:
        log_fn("[Segmenter] Không tìm thấy VnCoreNLP-1.1.1.jar → dùng underthesea")
        return None

    try:
        from vncorenlp import VnCoreNLP
        _segmenter_instance = VnCoreNLP(
            jar, annotators="wseg", max_heap_size="-Xmx500m"
        )
        log_fn(f"[Segmenter] RDRSegmenter sẵn sàng ← {jar}")
        return _segmenter_instance
    except ImportError:
        log_fn("[Segmenter] vncorenlp chưa cài → dùng underthesea")
    except Exception as e:
        log_fn(f"[Segmenter] Lỗi khởi tạo VnCoreNLP: {e} → dùng underthesea")

    return None


def segment_text(text: str, log_fn=print) -> str:
    """
    Tách từ tiếng Việt.
    Dùng singleton đã khởi tạo — không tạo lại JVM mỗi lần gọi.
    """
    global _underthesea_tokenize
    seg = get_segmenter(log_fn=log_fn)

    # Ưu tiên 1: RDRSegmenter
    if seg is not None:
        try:
            sents = seg.tokenize(text)
            return " ".join(" ".join(s) for s in sents)
        except Exception as e:
            log_fn(f"[Segmenter] Lỗi tokenize: {e}")

    # Ưu tiên 2: underthesea
    try:
        if _underthesea_tokenize is None:
            from underthesea import word_tokenize
            _underthesea_tokenize = word_tokenize
        return _underthesea_tokenize(text, format="text")
    except Exception:
        pass

    # Ưu tiên 3: giữ nguyên
    return text