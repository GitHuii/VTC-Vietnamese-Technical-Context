"""
ui/app.py
Giao diện Tkinter cho hệ thống đánh giá độ tương đồng ngữ nghĩa.
Chạy: python app.py  (từ thư mục gốc Code/)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Đảm bảo import được package core dù chạy từ thư mục con
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models    import create_model, MODEL_REGISTRY
from core.evaluator import evaluate_dataset, compute_correlations

# ── Hằng số giao diện
SCORE_COLS    = [f"s{i}" for i in range(1, 11)]
MODEL_CHOICES = list(MODEL_REGISTRY.keys())
ALL_COLUMNS   = (
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

        self.df_raw:    pd.DataFrame = None
        self.df_result: pd.DataFrame = None
        self._show_scores    = tk.BooleanVar(value=True)
        self._selected_model = tk.StringVar(value=MODEL_CHOICES[0])

        self._build_ui()

    # ══════════════════════════════════════════
    #  XÂY DỰNG GIAO DIỆN
    # ══════════════════════════════════════════

    def _build_ui(self):
        # ── Thanh điều khiển
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Mô hình:").pack(side="left", padx=4)
        ttk.Combobox(
            ctrl, textvariable=self._selected_model,
            values=MODEL_CHOICES, state="readonly", width=20
        ).pack(side="left", padx=4)

        ttk.Button(ctrl, text="📂 Tải file Excel", command=self._load_excel).pack(side="left", padx=8)
        ttk.Button(ctrl, text="▶ Chạy đánh giá",  command=self._run_eval ).pack(side="left", padx=4)
        ttk.Button(ctrl, text="💾 Xuất kết quả",  command=self._export   ).pack(side="left", padx=4)

        ttk.Checkbutton(
            ctrl, text="Hiển thị cột s1–s10",
            variable=self._show_scores,
            command=self._toggle_score_cols
        ).pack(side="left", padx=12)

        self._lbl_file = ttk.Label(ctrl, text="Chưa tải file", foreground="gray")
        self._lbl_file.pack(side="left", padx=8)

        # ── Hệ số tương quan
        stat = ttk.LabelFrame(self, text="Hệ số tương quan", padding=6)
        stat.pack(fill="x", padx=8, pady=2)

        self._lbl_pearson  = ttk.Label(stat, text="Pearson:  —")
        self._lbl_spearman = ttk.Label(stat, text="Spearman: —")
        self._lbl_pearson .pack(side="left", padx=16)
        self._lbl_spearman.pack(side="left", padx=16)

        # ── Bảng kết quả
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._tree = ttk.Treeview(tbl_frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        # ── Log console
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="x", padx=8, pady=4)

        self._log_text = tk.Text(
            log_frame, height=6, state="disabled",
            wrap="word", font=("Consolas", 9)
        )
        log_sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        self._log_text.pack(side="left", fill="x", expand=True)
        log_sb.pack(side="right", fill="y")

    # ══════════════════════════════════════════
    #  LOG
    # ══════════════════════════════════════════

    def _log(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.update_idletasks()

    # ══════════════════════════════════════════
    #  TẢI FILE EXCEL
    # ══════════════════════════════════════════

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

    # ══════════════════════════════════════════
    #  CHẠY ĐÁNH GIÁ (background thread)
    # ══════════════════════════════════════════

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

        threading.Thread(target=_worker, daemon=True).start()

    def _on_eval_done(self, df: pd.DataFrame):
        self._populate_table(df)
        self._update_stats(df)
        self._log("=== Đánh giá hoàn tất ===\n")

    # ══════════════════════════════════════════
    #  BẢNG KẾT QUẢ
    # ══════════════════════════════════════════

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
            **{f"s{i}": 48 for i in range(1, 11)},
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

    def _toggle_score_cols(self):
        df = self.df_result if self.df_result is not None else self.df_raw
        if df is not None:
            self._populate_table(df)

    # ══════════════════════════════════════════
    #  HỆ SỐ TƯƠNG QUAN
    # ══════════════════════════════════════════

    def _update_stats(self, df: pd.DataFrame):
        corr = compute_correlations(df)
        if corr is None:
            return
        self._lbl_pearson.configure(
            text=f"Pearson:  r = {corr['pearson_r']:.4f}  (p = {corr['pearson_p']:.4f})"
        )
        self._lbl_spearman.configure(
            text=f"Spearman: ρ = {corr['spearman_r']:.4f}  (p = {corr['spearman_p']:.4f})"
        )
        self._log(f"Pearson  r = {corr['pearson_r']:.4f} | Spearman ρ = {corr['spearman_r']:.4f}")

    # ══════════════════════════════════════════
    #  XUẤT KẾT QUẢ
    # ══════════════════════════════════════════

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
