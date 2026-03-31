"""
ui/app.py
Giao diện Tkinter (File main).
Cập nhật: Load toàn bộ dữ liệu, tự động kéo dãn cột context khi ẩn r1-r10.
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

SCORE_COLS    = [f"r{i}" for i in range(1, 11)]
MODEL_CHOICES = list(MODEL_REGISTRY.keys())
ALL_COLUMNS   = (
    ["ID", "word1", "pos1", "word2", "pos2", "context1", "context2"]
    + SCORE_COLS
    + ["avg_score", "model_score"]
)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hệ thống Đánh giá Độ Tương Đồng Ngữ Nghĩa")
        self.geometry("1300x750")
        self.resizable(True, True)

        self.df_current: pd.DataFrame = None
        self.file_status = "none" 
        
        self._show_scores    = tk.BooleanVar(value=True)
        self._selected_model = tk.StringVar(value=MODEL_CHOICES[0])
        
        self._build_ui()
        self._update_ui_states()

    def _build_ui(self):
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill="x")

        # Cụm 1: Tải & Chọn Mô hình
        self.btn_load = ttk.Button(ctrl, text="📂 Tải Dữ Liệu", command=self._load_file)
        self.btn_load.pack(side="left", padx=4)

        ttk.Label(ctrl, text="Mô hình:").pack(side="left", padx=4)
        self.cb_models = ttk.Combobox(ctrl, textvariable=self._selected_model, values=MODEL_CHOICES, state="readonly", width=16)
        self.cb_models.pack(side="left", padx=4)

        # Cụm 2: Các nút chức năng
        self.btn_eval = ttk.Button(ctrl, text="▶ Chạy Đánh Giá", command=self._run_eval)
        self.btn_eval.pack(side="left", padx=4)

        self.btn_export = ttk.Button(ctrl, text="💾 Xuất Kết Quả", command=self._export)
        self.btn_export.pack(side="left", padx=4)
        
        self.btn_plot = ttk.Button(ctrl, text="📊 Xem Biểu Đồ", command=self._show_plot_window)
        self.btn_plot.pack(side="left", padx=4)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        # Cụm 3: So sánh
        self.btn_compare = ttk.Button(ctrl, text="🏆 So Sánh Mô Hình", command=self._open_compare_dialog)
        self.btn_compare.pack(side="left", padx=4)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Checkbutton(
            ctrl, text="Hiển thị r1–r10",
            variable=self._show_scores,
            command=self._toggle_score_cols
        ).pack(side="left", padx=4)

        # Trạng thái File
        self._lbl_file = ttk.Label(ctrl, text="[Chưa tải file]", font=("Segoe UI", 10, "bold"), foreground="gray")
        self._lbl_file.pack(side="left", padx=8)

        # ── Hệ số tương quan
        stat = ttk.LabelFrame(self, text="Hệ số tương quan", padding=6)
        stat.pack(fill="x", padx=8, pady=2)
        self._lbl_pearson  = ttk.Label(stat, text="Pearson:  —", font=("Segoe UI", 10, "bold"))
        self._lbl_spearman = ttk.Label(stat, text="Spearman: —", font=("Segoe UI", 10, "bold"))
        self._lbl_pearson .pack(side="left", padx=16)
        self._lbl_spearman.pack(side="left", padx=16)

        # ── Bảng dữ liệu
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self._tree = ttk.Treeview(tbl_frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        # ── Log
        log_frame = ttk.LabelFrame(self, text="Log hệ thống", padding=4)
        log_frame.pack(fill="x", padx=8, pady=4)
        self._log_text = tk.Text(log_frame, height=5, state="disabled", font=("Consolas", 9))
        self._log_text.pack(fill="x")

    def _update_ui_states(self):
        if self.file_status == "none":
            self.btn_eval.config(state="disabled")
            self.btn_export.config(state="disabled")
            self.btn_plot.config(state="disabled")
            self.cb_models.config(state="normal")
            
        elif self.file_status == "raw":
            self.btn_eval.config(state="normal")     
            self.btn_export.config(state="disabled") 
            self.btn_plot.config(state="disabled")   
            self.cb_models.config(state="normal")
            
        elif self.file_status == "evaluated":
            self.btn_eval.config(state="disabled")   
            self.btn_export.config(state="disabled") 
            self.btn_plot.config(state="normal")     
            self.cb_models.config(state="disabled")
            
        elif self.file_status == "new_result":
            self.btn_eval.config(state="disabled")   
            self.btn_export.config(state="normal")   
            self.btn_plot.config(state="normal")     
            self.cb_models.config(state="disabled")

    def _log(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.update_idletasks()

    def _load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Dữ liệu", "*.xlsx *.xls *.csv")])
        if not path: return
        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.df_current = df
            
            if "model_score" in df.columns:
                self.file_status = "evaluated"
                self._lbl_file.configure(text=f"[KẾT QUẢ] {os.path.basename(path)}", foreground="green")
                self._log(f"Đã tải FILE KẾT QUẢ: {os.path.basename(path)}")
                self._update_stats(df)
            else:
                self.file_status = "raw"
                self._lbl_file.configure(text=f"[CHƯA ĐÁNH GIÁ] {os.path.basename(path)}", foreground="red")
                self._log(f"Đã tải FILE CHƯA ĐÁNH GIÁ: {os.path.basename(path)}")
                self._lbl_pearson.configure(text="Pearson:  —")
                self._lbl_spearman.configure(text="Spearman: —")

            self._populate_table(df)
            self._update_ui_states()

        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def _run_eval(self):
        model_name = self._selected_model.get()
        self._log(f"\n=== Bắt đầu đánh giá với: {model_name} ===")
        
        self.btn_eval.config(state="disabled")
        self.btn_load.config(state="disabled")
        
        threading.Thread(target=self._worker, args=(model_name,), daemon=True).start()

    def _worker(self, model_name):
        try:
            model = create_model(model_name)
            model.load(log_fn=self._log)
            df_out = evaluate_dataset(self.df_current, model, log_fn=self._log)
            
            self.df_current = df_out
            self.file_status = "new_result"
            
            self.after(0, lambda: self._on_eval_done())
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
            self.after(0, lambda: self._update_ui_states()) 
            self.after(0, lambda: self.btn_load.config(state="normal"))

    def _on_eval_done(self):
        self._lbl_file.configure(text="[ĐĐ ĐÁNH GIÁ XONG] Sẵn sàng xuất file", foreground="blue")
        self._populate_table(self.df_current)
        self._update_stats(self.df_current)
        self.btn_load.config(state="normal")
        self._update_ui_states()
        self._log("=== Đánh giá hoàn tất ===\n")

    def _export(self):
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            self.df_current.drop(columns=["raw_cosine"], errors="ignore").to_excel(path, index=False)
            messagebox.showinfo("Thành công", f"Đã lưu tại:\n{path}")
            self._log(f"Đã xuất file: {path}")

    def _show_plot_window(self):
        show_statistical_plots_window(self, self.df_current)

    def _open_compare_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Cấu hình So sánh Mô hình")
        dlg.geometry("550x300")
        dlg.transient(self)
        dlg.grab_set()

        paths = {"PhoBERT-Base": tk.StringVar(), "PhoBERT-Large": tk.StringVar(), "mBERT": tk.StringVar()}

        ttk.Label(dlg, text="Chọn file kết quả đã đánh giá cho từng mô hình (ít nhất 2):", font=("Segoe UI", 10, "bold")).pack(pady=10)

        def _browse(model_name):
            p = filedialog.askopenfilename(title=f"Chọn KQ {model_name}", filetypes=[("Dữ liệu", "*.xlsx *.csv")], parent=dlg)
            if p: paths[model_name].set(p)

        for model in ["PhoBERT-Base", "PhoBERT-Large", "mBERT"]:
            frame = ttk.Frame(dlg)
            frame.pack(fill="x", padx=15, pady=5)
            
            ttk.Label(frame, text=model, width=15, font=("Segoe UI", 9, "bold")).pack(side="left")
            ttk.Entry(frame, textvariable=paths[model], state="readonly", width=40).pack(side="left", padx=5)
            ttk.Button(frame, text="Chọn file", command=lambda m=model: _browse(m)).pack(side="left")

        def _draw():
            selected_models = {name: path.get() for name, path in paths.items() if path.get() != ""}
            if len(selected_models) < 1:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất 1 file kết quả.", parent=dlg)
                return
            
            results_dfs = {}
            for name, p in selected_models.items():
                try:
                    df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
                    if "model_score" not in df.columns:
                        messagebox.showerror("Lỗi", f"File của {name} không có cột 'model_score'!", parent=dlg)
                        return
                    results_dfs[name] = df
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không đọc được file {name}: {e}", parent=dlg)
                    return
            
            show_multi_model_comparison_plot(self, results_dfs)

        ttk.Button(dlg, text="📊 Chạy So Sánh & Vẽ Biểu Đồ", command=_draw, style="Accent.TButton").pack(pady=20)


    def _populate_table(self, df):
        cols = [c for c in ALL_COLUMNS if c in df.columns]
        
        # Kiểm tra nếu ẩn điểm r1-r10
        is_hiding_scores = not self._show_scores.get()
        if is_hiding_scores: 
            cols = [c for c in cols if c not in SCORE_COLS]
            
        self._tree.delete(*self._tree.get_children())
        self._tree["columns"] = cols
        
        # ─── THIẾT LẬP KÍCH THƯỚC CỘT LINH HOẠT ───
        for c in cols: 
            self._tree.heading(c, text=c)
            
            # Nếu là cột context, cho phép co dãn (stretch=True) và căn trái (anchor="w")
            if c in ["context1", "context2"]:
                # Nếu đang ẩn r1-r10 thì mở rộng cột context lên 400px, nếu không thì để 200px
                context_width = 400 if is_hiding_scores else 200
                self._tree.column(c, width=context_width, minwidth=150, stretch=True, anchor="w")
                
            # Cột từ vựng
            elif c in ["word1", "word2"]:
                self._tree.column(c, width=100, minwidth=80, stretch=False, anchor="center")
                
            # Cột ID, POS (Cột hẹp)
            elif c in ["ID", "pos1", "pos2"]:
                self._tree.column(c, width=50, minwidth=40, stretch=False, anchor="center")
                
            # Các cột điểm số
            else:
                self._tree.column(c, width=70, minwidth=60, stretch=False, anchor="center")

        # ─── LOAD TOÀN BỘ DỮ LIỆU (BỎ .head(1000)) ───
        for _, row in df.iterrows():
            self._tree.insert("", "end", values=[f"{row[c]:.4f}" if isinstance(row[c], float) else row[c] for c in cols])

    def _toggle_score_cols(self):
        df = self.df_current
        if df is not None: self._populate_table(df)

    def _update_stats(self, df):
        corr = compute_correlations(df)
        if corr:
            self._lbl_pearson.configure(text=f"Pearson:  r = {corr['pearson_r']:.4f}")
            self._lbl_spearman.configure(text=f"Spearman: ρ = {corr['spearman_r']:.4f}")
