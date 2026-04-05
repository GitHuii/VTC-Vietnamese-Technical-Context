"""
ui/app.py
Giao diện Tkinter — Light Research Dashboard Theme
Khóa luận tốt nghiệp: Đánh giá Độ Tương Đồng Ngữ Nghĩa
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models    import create_model, MODEL_REGISTRY
from core.evaluator import evaluate_dataset, compute_correlations
from ui.plotting    import show_statistical_plots_window, show_multi_model_comparison_plot

# ══════════════════════════════════════════════
#  PALETTE — Light Academic Research
# ══════════════════════════════════════════════
C_BG        = "#f5f7fa"
C_SURFACE   = "#ffffff"
C_SURFACE2  = "#eef1f6"
C_BORDER    = "#d4dae4"
C_BORDER2   = "#bec7d4"

C_NAVY      = "#1a2744"
C_BLUE      = "#2563eb"
C_BLUE_L    = "#dbeafe"
C_GREEN     = "#16a34a"
C_GREEN_L   = "#dcfce7"
C_AMBER     = "#d97706"
C_AMBER_L   = "#fef3c7"
C_RED       = "#dc2626"
C_PURPLE    = "#7c3aed"
C_PURPLE_L  = "#ede9fe"
C_ROSE      = "#be123c"
C_ROSE_L    = "#ffe4e6"

C_TEXT      = "#1e2736"
C_TEXT2     = "#4b5870"
C_TEXT3     = "#8594ab"
C_WHITE     = "#ffffff"

FONT_UI     = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI Semibold", 9)
FONT_TITLE  = ("Segoe UI Semibold", 12)
FONT_HEADER = ("Segoe UI Semibold", 10)
FONT_MONO   = ("Consolas", 9)
FONT_NUM    = ("Segoe UI Semibold", 16) # Giảm size xíu để vừa 9 ô
FONT_CAP    = ("Segoe UI", 8)

SCORE_COLS    = [f"r{i}" for i in range(1, 11)]
MODEL_CHOICES = list(MODEL_REGISTRY.keys())
ALL_COLUMNS   = (
    ["ID", "word1", "pos1", "word2", "pos2", "context1", "context2"]
    + SCORE_COLS
    + ["avg_score", "model_score"]
)

MODEL_COLORS = {
    "PhoBERT-Base":  (C_BLUE,   C_BLUE_L),
    "PhoBERT-Large": (C_PURPLE, C_PURPLE_L),
    "PhoBERT":       (C_BLUE,   C_BLUE_L),
    "mBERT":         (C_GREEN,  C_GREEN_L),
    "XMLRoBERTa":    (C_ROSE,   C_ROSE_L),
    "ELMo":          (C_AMBER,  C_AMBER_L),
}


# ══════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đo lường độ tương tự ngữ nghĩa thuật ngữ kỹ thuật theo ngữ cảnh")
        self.geometry("1440x860") # Mở rộng bề ngang để chứa thêm card
        self.minsize(1200, 720)
        self.resizable(True, True)
        self.configure(bg=C_BG)

        self.df_current: pd.DataFrame = None
        self.file_status = "none"
        self._show_scores    = tk.BooleanVar(value=True)
        self._selected_model = tk.StringVar(value=MODEL_CHOICES[0] if MODEL_CHOICES else "")

        self._apply_theme()
        self._build_ui()
        self._update_ui_states()

    # ──────────────────────────────────────────
    #  THEME
    # ──────────────────────────────────────────
    def _apply_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TFrame",      background=C_BG)
        s.configure("Card.TFrame", background=C_SURFACE, relief="flat")

        s.configure("TLabelframe",
                    background=C_SURFACE, bordercolor=C_BORDER,
                    relief="flat", padding=10)
        s.configure("TLabelframe.Label",
                    background=C_SURFACE, foreground=C_BLUE, font=FONT_BOLD)

        s.configure("TLabel",       background=C_BG,      foreground=C_TEXT,  font=FONT_UI)
        s.configure("H.TLabel",     background=C_SURFACE, foreground=C_TEXT,  font=FONT_BOLD)
        s.configure("Muted.TLabel", background=C_SURFACE, foreground=C_TEXT3, font=FONT_CAP)
        s.configure("Cap.TLabel",   background=C_SURFACE, foreground=C_TEXT3, font=FONT_CAP)
        s.configure("Num.TLabel",   background=C_SURFACE, foreground=C_BLUE,  font=FONT_NUM)
        s.configure("Green.TLabel", background=C_SURFACE, foreground=C_GREEN, font=FONT_NUM)
        s.configure("Amber.TLabel", background=C_SURFACE, foreground=C_AMBER, font=FONT_NUM)

        # Buttons
        s.configure("TButton",
                    background=C_SURFACE, foreground=C_TEXT,
                    bordercolor=C_BORDER, focuscolor=C_SURFACE,
                    font=FONT_UI, relief="flat", padding=(10, 6))
        s.map("TButton",
              background=[("active", C_SURFACE2), ("disabled", C_SURFACE2)],
              foreground=[("disabled", C_TEXT3)],
              bordercolor=[("active", C_BORDER2)])

        s.configure("Primary.TButton",
                    background=C_BLUE, foreground=C_WHITE,
                    bordercolor=C_BLUE, font=FONT_BOLD,
                    relief="flat", padding=(12, 6))
        s.map("Primary.TButton",
              background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")],
              foreground=[("disabled", C_WHITE)])

        s.configure("Success.TButton",
                    background=C_GREEN, foreground=C_WHITE,
                    bordercolor=C_GREEN, font=FONT_BOLD,
                    relief="flat", padding=(12, 6))
        s.map("Success.TButton",
              background=[("active", "#15803d"), ("disabled", "#86efac")],
              foreground=[("disabled", C_WHITE)])

        s.configure("Ghost.TButton",
                    background=C_BG, foreground=C_TEXT2,
                    bordercolor=C_BORDER, font=FONT_UI,
                    relief="flat", padding=(10, 6))
        s.map("Ghost.TButton",
              background=[("active", C_SURFACE2)],
              foreground=[("active", C_TEXT)])

        # Combobox
        s.configure("TCombobox",
                    fieldbackground=C_SURFACE, background=C_SURFACE,
                    foreground=C_TEXT, arrowcolor=C_TEXT2,
                    bordercolor=C_BORDER, font=FONT_UI)
        s.map("TCombobox",
              fieldbackground=[("readonly", C_SURFACE)],
              foreground=[("readonly", C_TEXT)],
              bordercolor=[("focus", C_BLUE)])

        # Checkbutton
        s.configure("TCheckbutton",
                    background=C_BG, foreground=C_TEXT2, font=FONT_UI)
        s.map("TCheckbutton",
              background=[("active", C_BG)],
              foreground=[("active", C_TEXT)])

        # Misc
        s.configure("TSeparator", background=C_BORDER)
        s.configure("TScrollbar",
                    background=C_SURFACE2, troughcolor=C_BG,
                    arrowcolor=C_TEXT3, bordercolor=C_BG, relief="flat")
        s.configure("Blue.Horizontal.TProgressbar",
                    troughcolor=C_SURFACE2, background=C_BLUE,
                    bordercolor=C_SURFACE2, lightcolor=C_BLUE, darkcolor=C_BLUE)

        # Treeview
        s.configure("Treeview",
                    background=C_SURFACE, foreground=C_TEXT,
                    fieldbackground=C_SURFACE, rowheight=26,
                    font=FONT_UI, bordercolor=C_BORDER, relief="flat")
        s.configure("Treeview.Heading",
                    background=C_SURFACE2, foreground=C_TEXT2,
                    font=FONT_BOLD, relief="flat", bordercolor=C_BORDER)
        s.map("Treeview",
              background=[("selected", C_BLUE_L)],
              foreground=[("selected", C_BLUE)])
        s.map("Treeview.Heading",
              background=[("active", C_BORDER)])

    # ──────────────────────────────────────────
    #  BUILD UI
    # ──────────────────────────────────────────
    def _build_ui(self):
        # ── HEADER ──────────────────────────────
        hdr = tk.Frame(self, bg=C_NAVY, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="◈", bg=C_NAVY, fg=C_BLUE,
                 font=("Segoe UI", 16)).pack(side="left", padx=(16, 4))
        tk.Label(hdr, text="Đo lường độ tương tự ngữ nghĩa thuật ngữ kỹ thuật theo ngữ cảnh",
                 bg=C_NAVY, fg=C_WHITE,
                 font=("Segoe UI Semibold", 13)).pack(side="left")
        tk.Label(hdr,
                 text="Nguyễn Viết Huy & Hoàng Thanh Chiến  ·  ĐHKT-KT Công Nghiệp",
                 bg=C_NAVY, fg="#6b8ec4",
                 font=("Segoe UI", 8)).pack(side="right", padx=16)

        # ── TOOLBAR ─────────────────────────────
        tb = tk.Frame(self, bg=C_SURFACE,
                      highlightbackground=C_BORDER, highlightthickness=1)
        tb.pack(fill="x")
        tb_i = tk.Frame(tb, bg=C_SURFACE)
        tb_i.pack(fill="x", padx=12, pady=6)

        self._make_tb_label(tb_i, "DỮ LIỆU")
        self.btn_load = ttk.Button(tb_i, text="📂  Tải file",
                                    command=self._load_file)
        self.btn_load.pack(side="left", padx=2)
        self._lbl_file = tk.Label(tb_i, text="Chưa tải file",
                                   bg=C_SURFACE, fg=C_TEXT3, font=FONT_UI)
        self._lbl_file.pack(side="left", padx=6)
        self._status_badge = tk.Label(tb_i, text="",
                                       bg=C_SURFACE, font=("Segoe UI Semibold", 8))
        self._status_badge.pack(side="left", padx=4)

        self._make_tb_sep(tb_i)
        self._make_tb_label(tb_i, "MÔ HÌNH")
        self.cb_models = ttk.Combobox(
            tb_i, textvariable=self._selected_model,
            values=MODEL_CHOICES, state="readonly", width=16
        )
        self.cb_models.pack(side="left", padx=2)
        self.cb_models.bind("<<ComboboxSelected>>", self._on_model_change)
        self._model_badge = tk.Label(tb_i, text=MODEL_CHOICES[0] if MODEL_CHOICES else "",
                                      bg=C_BLUE_L, fg=C_BLUE,
                                      font=("Segoe UI Semibold", 8),
                                      padx=6, pady=2)
        self._model_badge.pack(side="left", padx=4)

        self._make_tb_sep(tb_i)
        self._make_tb_label(tb_i, "HÀNH ĐỘNG")
        self.btn_eval = ttk.Button(tb_i, text="▶  Chạy đánh giá",
                                    style="Primary.TButton",
                                    command=self._run_eval)
        self.btn_eval.pack(side="left", padx=2)
        self.btn_export = ttk.Button(tb_i, text="💾  Xuất kết quả",
                                      style="Success.TButton",
                                      command=self._export)
        self.btn_export.pack(side="left", padx=2)
        self.btn_plot = ttk.Button(tb_i, text="📊  Biểu đồ",
                                    style="Ghost.TButton",
                                    command=self._show_plot_window)
        self.btn_plot.pack(side="left", padx=2)
        self.btn_compare = ttk.Button(tb_i, text="🏆  So sánh mô hình",
                                       style="Ghost.TButton",
                                       command=self._open_compare_dialog)
        self.btn_compare.pack(side="left", padx=2)

        self._make_tb_sep(tb_i)
        ttk.Checkbutton(tb_i, text="Hiện r1–r10",
                         variable=self._show_scores,
                         command=self._toggle_score_cols
                         ).pack(side="left", padx=6)

        # ── STAT CARDS (Grid Layout) ─────────────
        cards_wrap = tk.Frame(self, bg=C_BG)
        cards_wrap.pack(fill="x", padx=12, pady=(10, 6))

        # Khởi tạo các thẻ
        self._card_model    = self._make_card(cards_wrap, "Mô hình", MODEL_CHOICES[0] if MODEL_CHOICES else "", "H.TLabel")
        self._card_samples  = self._make_card(cards_wrap, "Tổng mẫu", "—", "Num.TLabel")
        
        self._card_pearson    = self._make_card(cards_wrap, "Pearson (r)", "—", "Green.TLabel")
        self._card_p_pearson  = self._make_card(cards_wrap, "p-value (P)", "—", "Muted.TLabel")
        self._card_mae        = self._make_card(cards_wrap, "MAE", "—", "Amber.TLabel")

        self._card_spearman   = self._make_card(cards_wrap, "Spearman (ρ)", "—", "Green.TLabel")
        self._card_p_spearman = self._make_card(cards_wrap, "p-value (S)", "—", "Muted.TLabel")
        self._card_rmse       = self._make_card(cards_wrap, "RMSE", "—", "Amber.TLabel")

        self._card_kendall    = self._make_card(cards_wrap, "Kendall (τ)", "—", "Green.TLabel")
        self._card_p_kendall  = self._make_card(cards_wrap, "p-value (K)", "—", "Muted.TLabel")
        self._card_r2         = self._make_card(cards_wrap, "R²", "—", "Green.TLabel")

        # Bố trí Grid cho thẻ (2 hàng)
        # Cột 0: Model | Cột 1: Samples | Cột 2: Pearson | Cột 3: Spearman | Cột 4: Kendall | Cột 5: Errors & R2
        self._card_model.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6), pady=2)
        self._card_samples.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 6), pady=2)

        self._card_pearson.grid(row=0, column=2, sticky="nsew", padx=(0, 6), pady=2)
        self._card_p_pearson.grid(row=1, column=2, sticky="nsew", padx=(0, 6), pady=2)

        self._card_spearman.grid(row=0, column=3, sticky="nsew", padx=(0, 6), pady=2)
        self._card_p_spearman.grid(row=1, column=3, sticky="nsew", padx=(0, 6), pady=2)

        self._card_kendall.grid(row=0, column=4, sticky="nsew", padx=(0, 6), pady=2)
        self._card_p_kendall.grid(row=1, column=4, sticky="nsew", padx=(0, 6), pady=2)

        self._card_mae.grid(row=0, column=5, sticky="nsew", padx=(0, 6), pady=2)
        self._card_rmse.grid(row=1, column=5, sticky="nsew", padx=(0, 6), pady=2)

        self._card_r2.grid(row=0, column=6, rowspan=2, sticky="nsew", padx=(0, 6), pady=2)

        # Cho phép các cột co giãn đều
        for i in range(7):
            cards_wrap.columnconfigure(i, weight=1)

        # ── PROGRESS ────────────────────────────
        self._prog_frame = tk.Frame(self, bg=C_BG)
        self._prog_frame.pack(fill="x", padx=12, pady=(0, 4))
        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(
            self._prog_frame, variable=self._prog_var,
            maximum=100, mode="determinate",
            style="Blue.Horizontal.TProgressbar"
        )
        self._prog_lbl = tk.Label(self._prog_frame, text="",
                                   bg=C_BG, fg=C_TEXT2, font=FONT_CAP)

        # ── TABLE ───────────────────────────────
        tbl_outer = tk.Frame(self, bg=C_BORDER, bd=1)
        tbl_outer.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        tbl_frame = tk.Frame(tbl_outer, bg=C_SURFACE)
        tbl_frame.pack(fill="both", expand=True)

        self._tree = ttk.Treeview(tbl_frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)
        self._tree.tag_configure("odd",  background=C_SURFACE)
        self._tree.tag_configure("even", background="#f8fafc")

        # ── LOG ─────────────────────────────────
        log_wrap = tk.Frame(self, bg=C_BORDER, bd=1)
        log_wrap.pack(fill="x", padx=12, pady=(0, 10))
        log_head = tk.Frame(log_wrap, bg=C_SURFACE2, height=22)
        log_head.pack(fill="x")
        log_head.pack_propagate(False)
        tk.Label(log_head, text="  ▸  Console",
                 bg=C_SURFACE2, fg=C_TEXT3,
                 font=("Segoe UI", 8)).pack(side="left", pady=2)

        self._log_text = tk.Text(
            log_wrap, height=5,
            state="disabled", wrap="word",
            font=FONT_MONO,
            bg="#1e2736", fg="#a8d8a0",
            insertbackground=C_BLUE,
            selectbackground=C_BLUE_L,
            relief="flat", bd=0,
            padx=10, pady=6
        )
        log_sb = ttk.Scrollbar(log_wrap, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        self._log_text.pack(side="left", fill="x", expand=True)
        log_sb.pack(side="right", fill="y")

    # ──────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────
    def _make_tb_label(self, parent, text):
        tk.Label(parent, text=text, bg=C_SURFACE,
                 fg=C_TEXT3, font=("Segoe UI", 7)).pack(side="left", padx=(8, 2))

    def _make_tb_sep(self, parent):
        tk.Frame(parent, bg=C_BORDER, width=1, height=22).pack(
            side="left", padx=8, pady=4)

    def _make_card(self, parent, caption, value, val_style):
        frame = tk.Frame(parent, bg=C_SURFACE,
                         highlightbackground=C_BORDER, highlightthickness=1)
        frame.configure(padx=16, pady=6)
        tk.Label(frame, text=caption, bg=C_SURFACE,
                 fg=C_TEXT3, font=FONT_CAP).pack(anchor="w")
        val = ttk.Label(frame, text=value, style=val_style)
        val.pack(anchor="w")
        frame._val = val
        return frame

    def _set_card(self, card, text):
        card._val.configure(text=text)

    def _on_model_change(self, _=None):
        name = self._selected_model.get()
        self._set_card(self._card_model, name)
        fg, bg = MODEL_COLORS.get(name, (C_BLUE, C_BLUE_L))
        self._model_badge.configure(text=name, bg=bg, fg=fg)

    # ──────────────────────────────────────────
    #  STATE MACHINE
    # ──────────────────────────────────────────
    def _update_ui_states(self):
        s = self.file_status
        self.btn_eval   .config(state="normal"   if s == "raw"                        else "disabled")
        self.btn_export .config(state="normal"   if s == "new_result"                 else "disabled")
        self.btn_plot   .config(state="normal"   if s in ("evaluated", "new_result")  else "disabled")
        self.cb_models  .config(state="readonly" if s in ("none", "raw")              else "disabled")

        text_map = {
            "none":       ("", C_TEXT3, C_SURFACE),
            "raw":        ("Chưa đánh giá", C_AMBER, C_AMBER_L),
            "evaluated":  ("Đã có kết quả", C_GREEN, C_GREEN_L),
            "new_result": ("Đánh giá xong ✓", C_GREEN, C_GREEN_L),
        }
        t, fg, bg = text_map.get(s, ("", C_TEXT3, C_SURFACE))
        self._status_badge.configure(
            text=f"  {t}  " if t else "",
            bg=bg, fg=fg,
            font=("Segoe UI Semibold", 8)
        )

    # ──────────────────────────────────────────
    #  LOG
    # ──────────────────────────────────────────
    def _log(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.update_idletasks()

    # ──────────────────────────────────────────
    #  LOAD FILE
    # ──────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Dữ liệu", "*.xlsx *.xls *.csv")]
        )
        if not path:
            return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.df_current = df
            fname = os.path.basename(path)
            n = len(df)

            if "model_score" in df.columns:
                self.file_status = "evaluated"
                self._lbl_file.configure(text=fname, fg=C_TEXT)
                self._log(f"→ FILE KẾT QUẢ: {fname}  ({n:,} mẫu)")
                self._update_stats(df)
            else:
                self.file_status = "raw"
                self._lbl_file.configure(text=fname, fg=C_TEXT)
                self._log(f"→ File dữ liệu: {fname}  ({n:,} mẫu)")
                # Reset metrics
                for c in [self._card_pearson, self._card_p_pearson, self._card_mae,
                          self._card_spearman, self._card_p_spearman, self._card_rmse,
                          self._card_kendall, self._card_p_kendall, self._card_r2]:
                    self._set_card(c, "—")

            self._set_card(self._card_samples, f"{n:,}")
            self._populate_table(df)
            self._update_ui_states()

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    # ──────────────────────────────────────────
    #  RUN EVAL
    # ──────────────────────────────────────────
    def _run_eval(self):
        model_name = self._selected_model.get()
        self._log(f"\n{'─'*44}")
        self._log(f"▶  Bắt đầu đánh giá · {model_name}")
        self._log(f"{'─'*44}")
        self.btn_eval.config(state="disabled")
        self.btn_load.config(state="disabled")
        self._show_progress(True)
        threading.Thread(target=self._worker, args=(model_name,), daemon=True).start()

    def _worker(self, model_name):
        import re

        def _log_prog(msg):
            self._log(msg)
            m = re.search(r"Tiến độ:\s*(\d+)/(\d+)", msg)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                pct = cur / tot * 100 if tot else 0
                self._prog_var.set(pct)
                self._prog_lbl.configure(text=f"{cur}/{tot}  ({pct:.0f}%)")

        try:
            model = create_model(model_name)
            model.load(log_fn=_log_prog)
            df_out = evaluate_dataset(self.df_current, model, log_fn=_log_prog)
            self.df_current = df_out
            self.file_status = "new_result"
            self.after(0, self._on_eval_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            self.after(0, lambda: self._log(f"[ERR] {e}"))
            self.after(0, self._update_ui_states)
            self.after(0, lambda: self.btn_load.config(state="normal"))
        finally:
            self.after(0, lambda: self._show_progress(False))

    def _show_progress(self, show: bool):
        if show:
            self._prog_bar.pack(side="left", fill="x", expand=True)
            self._prog_lbl.pack(side="left", padx=6)
        else:
            self._prog_bar.pack_forget()
            self._prog_lbl.pack_forget()
            self._prog_var.set(0)

    def _on_eval_done(self):
        self._lbl_file.configure(fg=C_GREEN)
        self._populate_table(self.df_current)
        self._update_stats(self.df_current)
        self.btn_load.config(state="normal")
        self._update_ui_states()
        self._log("─" * 44)
        self._log("✓  Đánh giá hoàn tất\n")

    # ──────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────
    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if path:
            self.df_current.drop(columns=["raw_cosine"], errors="ignore").to_excel(path, index=False)
            self._log(f"→ Đã xuất: {path}")
            messagebox.showinfo("Thành công", f"Đã lưu tại:\n{path}")

    # ──────────────────────────────────────────
    #  PLOTS
    # ──────────────────────────────────────────
    def _show_plot_window(self):
        show_statistical_plots_window(self, self.df_current)

    def _open_compare_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("So sánh mô hình")
        dlg.geometry("620x360")
        dlg.configure(bg=C_BG)
        dlg.transient(self)
        dlg.grab_set()

        # Dialog header
        hdr = tk.Frame(dlg, bg=C_NAVY, height=40)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  🏆  So sánh đa mô hình",
                 bg=C_NAVY, fg=C_WHITE,
                 font=("Segoe UI Semibold", 11)).pack(side="left", padx=12, pady=8)

        content = tk.Frame(dlg, bg=C_BG)
        content.pack(fill="both", expand=True, padx=20, pady=12)

        tk.Label(content,
                 text="Chọn file kết quả đã đánh giá cho từng mô hình (ít nhất 1 file)",
                 bg=C_BG, fg=C_TEXT2, font=FONT_UI).grid(row=0, column=0, columnspan=3, pady=(0, 16), sticky="w")

        paths = {m: tk.StringVar() for m in ["PhoBERT", "mBERT", "XMLRoBERTa"]}

        def _browse(model_name):
            p = filedialog.askopenfilename(
                title=f"Chọn KQ {model_name}",
                filetypes=[("Dữ liệu", "*.xlsx *.csv")], parent=dlg
            )
            if p:
                paths[model_name].set(p)

        # Sử dụng Grid Layout cho thẳng hàng
        for i, model in enumerate(["PhoBERT", "mBERT", "XMLRoBERTa"]):
            fg, bg = MODEL_COLORS.get(model, (C_BLUE, C_BLUE_L))
            
            lbl = tk.Label(content, text=f"  {model}  ", bg=bg, fg=fg,
                           font=("Segoe UI Semibold", 8), padx=4, pady=2)
            lbl.grid(row=i+1, column=0, pady=6, sticky="w")
            
            ent = ttk.Entry(content, textvariable=paths[model], state="readonly", width=45)
            ent.grid(row=i+1, column=1, padx=10, pady=6, sticky="we")
            
            btn = ttk.Button(content, text="Chọn", style="Ghost.TButton",
                             command=lambda m=model: _browse(m))
            btn.grid(row=i+1, column=2, pady=6)

        content.columnconfigure(1, weight=1)

        def _draw():
            selected = {n: p.get() for n, p in paths.items() if p.get()}
            if not selected:
                messagebox.showwarning("Cảnh báo", "Chọn ít nhất 1 file.", parent=dlg)
                return
            results_dfs = {}
            for name, p in selected.items():
                try:
                    df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
                    if "model_score" not in df.columns:
                        messagebox.showerror("Lỗi", f"File {name} thiếu cột 'model_score'!", parent=dlg)
                        return
                    results_dfs[name] = df
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Lỗi đọc file {name}: {e}", parent=dlg)
                    return
            
            # Đóng dialog trước khi gọi đồ thị để tránh lỗi hiển thị chồng lấp Matplotlib
            dlg.destroy()
            show_multi_model_comparison_plot(self, results_dfs)

        ttk.Button(content, text="📊  Chạy so sánh & vẽ biểu đồ",
                   style="Primary.TButton", command=_draw).grid(row=4, column=0, columnspan=3, pady=(24, 0))

    # ──────────────────────────────────────────
    #  TABLE
    # ──────────────────────────────────────────
    def _populate_table(self, df):
        cols = [c for c in ALL_COLUMNS if c in df.columns]
        hide = not self._show_scores.get()
        if hide:
            cols = [c for c in cols if c not in SCORE_COLS]

        self._tree.delete(*self._tree.get_children())
        self._tree["columns"] = cols

        for c in cols:
            self._tree.heading(c, text=c, anchor="center")
            if c in ("context1", "context2"):
                w = 380 if hide else 200
                self._tree.column(c, width=w, minwidth=150, stretch=True, anchor="w")
            elif c in ("word1", "word2"):
                self._tree.column(c, width=100, minwidth=80,  stretch=False, anchor="center")
            elif c in ("ID", "pos1", "pos2"):
                self._tree.column(c, width=50,  minwidth=40,  stretch=False, anchor="center")
            else:
                self._tree.column(c, width=72,  minwidth=60,  stretch=False, anchor="center")

        for i, (_, row) in enumerate(df.iterrows()):
            tag = "even" if i % 2 == 0 else "odd"
            vals = [
                f"{row[c]:.4f}" if isinstance(row[c], float) else row[c]
                for c in cols
            ]
            self._tree.insert("", "end", values=vals, tags=(tag,))

    def _toggle_score_cols(self):
        if self.df_current is not None:
            self._populate_table(self.df_current)

    # ──────────────────────────────────────────
    #  STATS
    # ──────────────────────────────────────────
    def _update_stats(self, df):
        corr = compute_correlations(df)
        if corr:
            self._set_card(self._card_pearson,   f"{corr['pearson_r']}")
            self._set_card(self._card_p_pearson, f"{corr['pearson_p']}")
            self._set_card(self._card_mae,       f"{corr['mae']}")
            
            self._set_card(self._card_spearman,   f"{corr['spearman_r']}")
            self._set_card(self._card_p_spearman, f"{corr['spearman_p']}")
            self._set_card(self._card_rmse,       f"{corr['rmse']}")
            
            self._set_card(self._card_kendall,   f"{corr['kendall_tau']}")
            self._set_card(self._card_p_kendall, f"{corr['kendall_p']}")
            self._set_card(self._card_r2,        f"{corr['r2']}")
            
            self._set_card(self._card_samples,   f"{corr['n_valid']}")