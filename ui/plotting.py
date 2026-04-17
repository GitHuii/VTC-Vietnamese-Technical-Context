"""
ui/plotting.py
Module chứa code vẽ biểu đồ cho hệ thống đánh giá.
Đã xử lý vẽ đa cửa sổ/đa mô hình.
"""

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, gaussian_kde, pearsonr, spearmanr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def show_statistical_plots_window(master_win: tk.Tk, df_result: pd.DataFrame):
    """(Nút Xem Biểu Đồ Đơn) - Hiện 2 cửa sổ cho 1 model."""
    valid_df = df_result[["avg_score", "model_score"]].dropna()
    if len(valid_df) < 2: return

    # CỬA SỔ 1: Mật độ
    win1 = tk.Toplevel(master_win)
    win1.title("Biểu Đồ Mật độ Tương đồng")
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=100)
    _draw_scatter_density(ax1, valid_df["avg_score"].values, valid_df["model_score"].values, "Biểu Đồ Mật độ Tương Đồng")
    _embed_plot_to_tkinter(win1, fig1)

    # CỬA SỔ 2: Cột
    win2 = tk.Toplevel(master_win)
    win2.title("Biểu Đồ Phân Bố Điểm")
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=100)
    _draw_bar_distribution(ax2, valid_df["model_score"].values, "Phân Bố Điểm Đánh Giá")
    _embed_plot_to_tkinter(win2, fig2)


def show_multi_model_comparison_plot(master_win: tk.Tk, results_dfs: dict):
    """
    (Nút So Sánh) - Hiển thị 3 cửa sổ:
      1. Scatter Heatmaps (chung cửa sổ, nhiều subplots)
      2. Bar Distributions (chung cửa sổ, nhiều subplots)
      3. Pearson & Spearman Bar Chart (chung cửa sổ)
    """
    models = list(results_dfs.keys())
    n = len(models)

    # ════ CỬA SỔ 1: SCATTER MẬT ĐỘ ════
    win_scatter = tk.Toplevel(master_win)
    win_scatter.title("So sánh: Biểu đồ Mật độ Tương đồng")
    fig_scat, axes_scat = plt.subplots(1, n, figsize=(5 * n, 5), dpi=100)
    if n == 1: axes_scat = [axes_scat] # Normalize to list

    for i, model in enumerate(models):
        df = results_dfs[model].dropna(subset=["avg_score", "model_score"])
        _draw_scatter_density(axes_scat[i], df["avg_score"].values, df["model_score"].values, f"Mật độ: {model}")
    
    plt.tight_layout()
    _embed_plot_to_tkinter(win_scatter, fig_scat)

    # ════ CỬA SỔ 2: PHÂN BỐ ĐIỂM (BAR) ════
    win_bar = tk.Toplevel(master_win)
    win_bar.title("So sánh: Phân bố số lượng điểm")
    fig_bar, axes_bar = plt.subplots(1, n, figsize=(5 * n, 5), dpi=100)
    if n == 1: axes_bar = [axes_bar]

    for i, model in enumerate(models):
        df = results_dfs[model].dropna(subset=["model_score"])
        _draw_bar_distribution(axes_bar[i], df["model_score"].values, f"Phân bố: {model}")
    
    plt.tight_layout()
    _embed_plot_to_tkinter(win_bar, fig_bar)

    # ════ CỬA SỔ 3: PEARSON & SPEARMAN ════
    pearson_scores = []
    spearman_scores = []
    
    for model in models:
        df = results_dfs[model].dropna(subset=["avg_score", "model_score"])
        if len(df) > 1:
            p_r, _ = pearsonr(df["avg_score"], df["model_score"])
            s_r, _ = spearmanr(df["avg_score"], df["model_score"])
            pearson_scores.append(p_r)
            spearman_scores.append(s_r)
        else:
            pearson_scores.append(0)
            spearman_scores.append(0)

    win_corr = tk.Toplevel(master_win)
    win_corr.title("So sánh Hệ số Tương quan")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6), dpi=100)
    
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax_corr.bar(x - width/2, pearson_scores, width, label='Pearson', color='#4CAF50')
    rects2 = ax_corr.bar(x + width/2, spearman_scores, width, label='Spearman', color='#2196F3')

    ax_corr.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax_corr.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)

    ax_corr.set_title('So sánh Pearson và Spearman giữa các Mô hình', fontsize=12, fontweight='bold')
    ax_corr.set_ylabel('Hệ số tương quan (0 - 1)', fontsize=10)
    ax_corr.set_xticks(x)
    ax_corr.set_xticklabels(models, fontsize=10)
    ax_corr.set_ylim(0, 1.1)
    ax_corr.legend(loc='upper left')
    ax_corr.grid(True, axis='y', linestyle='--', alpha=0.6)

    _embed_plot_to_tkinter(win_corr, fig_corr)


# ─── CÁC HÀM VẼ PHỤ TRỢ (Dùng chung để tránh lặp code) ───

def _draw_scatter_density(ax, x, y, title):
    if len(x) < 2: return
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        scatter = ax.scatter(x, y, c=z, s=15, cmap='viridis', edgecolor=None)
    except:
        scatter = ax.scatter(x, y, color='blue', s=15, alpha=0.5)

    slope, intercept, r, _, _ = linregress(x, y)
    line = slope * x + intercept

    ax.plot(x, line, color='red', linestyle='-', linewidth=2, label=f'r = {r:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("avg_score", fontsize=9)
    ax.set_ylabel("model_score", fontsize=9)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)


def _draw_bar_distribution(ax, scores, title):
    bins   = [0, 3.0, 6.0, 8.0, 10.01] 
    labels = ["0-3", "3.1-6", "6.1-8", "8.1-10"]
    colors = ['#f44336', '#ff9800', '#ffeb3b', '#4caf50'] # Đỏ, Cam, Vàng, Xanh

    counts = pd.cut(scores, bins=bins, labels=labels, right=False).value_counts().sort_index()
    x_bar = np.arange(len(counts))
    y_bar = counts.values
    
    ax.bar(x_bar, y_bar, color=colors, edgecolor='gray', width=0.6)
    for i, count in enumerate(y_bar):
        ax.text(i, count + (y_bar.max() * 0.02), f"{count}", ha='center', va='bottom', fontsize=9)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x_bar)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(y_bar) * 1.15 if len(y_bar) > 0 else 10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)


def _embed_plot_to_tkinter(win, fig):
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, win)
    toolbar.update()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)