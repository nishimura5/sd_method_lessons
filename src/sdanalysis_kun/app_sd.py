import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .sd_funcs import (
    compute_cronbach_alpha,
    factor_analysis,
    get_japanese_monospace_font,
    pickup_by_factor,
    print_parallel_analysis_summary,
    set_japanese_font,
    summarize_factor_scores,
)
from .sd_plot import create_factor_map_figure, create_pca_map_figure, plot_factor_loadings
from .stimulus_images import find_stimulus_png, find_thumbnail_png_folder
from .tooltip import ToolTip


def _read_sd_csv(path, encoding):
    """CSVを読み込み、stimulus_idの表記と文字列型をそのまま保持する。"""
    return pd.read_csv(path, encoding=encoding, converters={"stimulus_id": str})


class SDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SDAnalysis-kun")
        self.root.geometry("1400x700")
        self.root.minsize(700, 400)

        self.df = None
        self.check_vars = {}
        self.pattern_loading_df = None
        self.structure_loading_df = None
        self.score_df = None
        self.score_summary_df = None
        self.factor_names = None
        self.factor_corr_df = None
        self.filtered_df = None
        self.target_stimulus_table = {}  # 分析対象の刺激名のホワイトリスト、{"colname": [stimulus1, stimulus2, ...], ...} の形式
        self.invert_map = {}
        self.png_folder_var = tk.StringVar(value="")

        set_japanese_font()
        self._build_ui()

    def _build_ui(self):
        # === CSVファイル・PNGフォルダ選択 ===
        frame_source = ttk.Frame(self.root)
        frame_source.pack(fill=tk.X, padx=10, pady=(10, 5))

        frame_file = ttk.LabelFrame(frame_source, text="Select CSV File", padding=10)
        frame_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.file_path_var = tk.StringVar()
        ttk.Entry(frame_file, textvariable=self.file_path_var, state="readonly", width=70).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(frame_file, text="Browse", command=self._select_file).pack(side=tk.LEFT)

        self._add_png_folder_control(frame_source)

        # === 刺激名カラム選択 ===
        frame_row = ttk.Frame(self.root)
        frame_row.pack(fill=tk.X, padx=10, pady=5)

        frame_stimulus = ttk.LabelFrame(frame_row, text="Stimulus Column", padding=10)
        frame_stimulus.pack(side=tk.LEFT, fill=tk.X, padx=(5, 2))

        self.stimulus_col_var = tk.StringVar()
        self.stimulus_col_combo = ttk.Combobox(
            frame_stimulus, textvariable=self.stimulus_col_var, state="readonly", width=16
        )
        self.stimulus_col_combo.pack(side=tk.LEFT)

        # === 刺激名フィルターダイアログを開くボタン ===
        ttk.Button(frame_stimulus, text="Filter...", command=self._open_stimulus_filter_dialog).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # === 回答者名カラム選択（任意） ===
        frame_resp = ttk.LabelFrame(frame_row, text="Respondent Column (optional)", padding=10)
        frame_resp.pack(side=tk.LEFT, fill=tk.X, padx=(0, 5))

        self.resp_col_var = tk.StringVar(value="")
        self.resp_col_combo = ttk.Combobox(frame_resp, textvariable=self.resp_col_var, state="readonly", width=16)
        self.resp_col_combo.pack(side=tk.LEFT)

        # === 形容詞対名の正規表現編集 ===
        frame_regex = ttk.LabelFrame(frame_row, text="Adjective Pair Regex (optional)", padding=10)
        frame_regex.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.regex_var = tk.StringVar(value="")
        regex_entry = ttk.Entry(frame_regex, textvariable=self.regex_var, width=20)
        regex_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ToolTip(
            regex_entry,
            '[Example]\n  Original: "Q1_warm_cold"\n  Regex: Q\\d+_(.+)_(.+)\n  Result: warm - cold',
        )
        self.btn_apply_reg = ttk.Button(frame_regex, text="Apply", command=self._apply_regex, state=tk.DISABLED)
        self.btn_apply_reg.pack(side=tk.LEFT)

        ttk.Label(frame_regex, text="Scale:").pack(side=tk.LEFT, padx=(5, 2))
        self.scale_var = tk.StringVar(value="7")
        ttk.Combobox(frame_regex, textvariable=self.scale_var, state="readonly", values=["5", "7"], width=3).pack(
            side=tk.LEFT,
        )

        self.btn_run_analysis = ttk.Button(frame_row, text="Run Analysis", command=self._run_analysis, width=14)
        self.btn_run_analysis.pack(side=tk.LEFT, padx=(5, 0), fill=tk.Y)

        # === 左右分割: 形容詞対カラム選択（左）と結果表示（右） ===
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, height=360)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- 左側: 形容詞対カラム選択 ---
        frame_adj = ttk.LabelFrame(paned, text="Select Adjective Pair Columns", padding=10)
        paned.add(frame_adj, weight=4)

        # 形容詞対が多い場合に備え、チェックボックス一覧を縦スクロール可能にする
        self.adjective_canvas = tk.Canvas(frame_adj, highlightthickness=0)
        # macOS の ttk.Scrollbar は Aqua の設定によって自動的に隠れるため、
        # 常に表示されるクラシックウィジェットを使う。
        adjective_scrollbar = tk.Scrollbar(
            frame_adj,
            orient=tk.VERTICAL,
            command=self.adjective_canvas.yview,
            width=16,
        )
        self.check_frame = ttk.Frame(self.adjective_canvas)
        self._adjective_window = self.adjective_canvas.create_window(
            (0, 0),
            window=self.check_frame,
            anchor="nw",
        )

        self.check_frame.bind("<Configure>", self._update_adjective_scrollregion)
        self.adjective_canvas.bind("<Configure>", self._resize_adjective_check_frame)
        self._bind_adjective_mousewheel(self.adjective_canvas)
        self._bind_adjective_mousewheel(self.check_frame)
        self.adjective_canvas.configure(yscrollcommand=adjective_scrollbar.set)

        # 先にスクロールバーの幅を確保し、残りを一覧領域に割り当てる。
        adjective_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.adjective_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 中央左: 併行分析表示 ---
        frame_parallel = ttk.LabelFrame(paned, text="Parallel Analysis", padding=10)
        paned.add(frame_parallel, weight=3)
        frame_corr = ttk.Frame(frame_parallel)
        frame_corr.pack(fill=tk.X, pady=(2, 7))
        # pearsonとpolychoricを選択するcomboを追加
        ttk.Label(frame_corr, text="Correlation:").pack(side=tk.LEFT, padx=(0, 2))
        self.corr_name_var = tk.StringVar(value="pearson")
        corr_combo = ttk.Combobox(
            frame_corr,
            textvariable=self.corr_name_var,
            state="readonly",
            values=["pearson", "polychoric"],
            width=10,
        )
        corr_combo.pack(side=tk.LEFT)
        corr_combo.bind("<<ComboboxSelected>>", self._on_corr_change)
        self.parallel_analysis_iter_var = tk.StringVar(value="500")
        ttk.Label(frame_corr, text="Iter.:").pack(side=tk.LEFT, padx=(5, 2))
        self.polychoric_entry = ttk.Entry(
            frame_corr, textvariable=self.parallel_analysis_iter_var, width=8, state="readonly"
        )
        self.polychoric_entry.pack(side=tk.LEFT, padx=(5, 0))

        frame_progress = ttk.Frame(frame_parallel)
        frame_progress.pack(fill=tk.X, pady=(0, 7))
        self.parallel_progress_label_var = tk.StringVar(value="")
        ttk.Label(frame_progress, textvariable=self.parallel_progress_label_var, width=20).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.parallel_progress_var = tk.DoubleVar(value=0)
        self.parallel_progress = ttk.Progressbar(
            frame_progress,
            variable=self.parallel_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.parallel_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.parallel_text = tk.Text(frame_parallel, wrap=tk.NONE, font=(get_japanese_monospace_font(), 11))
        parallel_scroll_y = ttk.Scrollbar(frame_parallel, orient=tk.VERTICAL, command=self.parallel_text.yview)
        self.parallel_text.configure(yscrollcommand=parallel_scroll_y.set)
        parallel_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.parallel_text.pack(fill=tk.BOTH, expand=True)
        ToolTip(
            self.parallel_text,
            "Parallel analysis compares eigenvalues from your data with those from random data.\n"
            'Selecting "PA" for Factors uses the number of factors suggested by this method.\n\n'
            "Legend: F=Factor, Obs=Observed, Rnd95=Random 95th percentile, Dif=Obs-Rnd95, Ret=Retained (Y/N).",
            position="bottom",
        )

        # --- 中央右: treeviewで各形容詞対のmeanとstdを表示 ---
        frame_center = ttk.LabelFrame(paned, text="Adjective Pair Statistics", padding=10)
        paned.add(frame_center, weight=6)

        def _init_sash(event, p=paned, _done=[False]):
            if not _done[0] and p.winfo_width() > 1:
                _done[0] = True
                p.sashpos(0, int(p.winfo_width() * 0.21))
                p.sashpos(1, int(p.winfo_width() * 0.48))

        paned.bind("<Configure>", _init_sash)

        # 因子数選択と実行
        frame_exec = ttk.Frame(frame_center)
        frame_exec.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(frame_exec, text="Factors:").pack(side=tk.LEFT, padx=(0, 2))
        self.n_factors_var = tk.StringVar(value="3")
        self.n_factors_combo = ttk.Combobox(
            frame_exec,
            textvariable=self.n_factors_var,
            state="readonly",
            values=["1", "2", "3", "4", "5", "PA"],
            width=4,
        )
        self.n_factors_combo.pack(side=tk.LEFT, padx=(0, 10))
        # defaultでは3を選択
        self.n_factors_combo.current(2)

        ttk.Label(frame_exec, text="Rotation:").pack(side=tk.LEFT, padx=(0, 2))
        self.rotation_var = tk.StringVar(value="varimax")
        self.rotation_combo = ttk.Combobox(
            frame_exec,
            textvariable=self.rotation_var,
            state="readonly",
            values=["promax", "varimax", "No rotation"],
            width=10,
        )
        self.rotation_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.rotation_combo.current(0)
        self.current_rotation = self.rotation_var.get()

        self.btn_plot_loadings = ttk.Button(
            frame_exec, text="Plot Loadings", command=self._plot_loadings, state=tk.DISABLED, width=14
        )
        self.btn_plot_loadings.pack(side=tk.LEFT, padx=(5, 5))
        self.btn_export_loadings = ttk.Button(
            frame_exec, text="Export Loadings", command=self._export_loadings_csv, state=tk.DISABLED, width=16
        )
        self.btn_export_loadings.pack(side=tk.LEFT)

        cols = ("mean", "std")
        self.stats_tree = ttk.Treeview(frame_center, columns=cols, show="tree headings", selectmode="browse")
        self.stats_tree.heading("#0", text="Adjective Pair", anchor=tk.W)
        self.stats_tree.heading("mean", text="Mean", anchor=tk.CENTER)
        self.stats_tree.heading("std", text="Std", anchor=tk.CENTER)
        self.stats_tree.column("#0", width=180, stretch=True)
        self.stats_tree.column("mean", width=70, anchor=tk.CENTER, stretch=False)
        self.stats_tree.column("std", width=70, anchor=tk.CENTER, stretch=False)

        stats_scroll = ttk.Scrollbar(frame_center, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 下部: 結果表示 ---
        frame_bottom = ttk.Frame(self.root)
        frame_bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        frame_bottom.columnconfigure(0, weight=1)
        frame_bottom.rowconfigure(0, weight=1)

        frame_bottom_right = ttk.LabelFrame(frame_bottom, text="Factor Score Summary (Mean / SD)", padding=10)
        frame_bottom_right.grid(row=0, column=0, sticky="nsew")
        frame_bottom_right.grid_propagate(False)

        # グラフ描画ボタン
        frame_plot = ttk.Frame(frame_bottom_right)
        frame_plot.pack(fill=tk.X, pady=(0, 5))

        self.btn_plot_map = ttk.Button(
            frame_plot,
            text="Plot Stimulus Map",
            command=self._plot_stimulus_map,
            state=tk.DISABLED,
        )
        self.btn_plot_map.pack(side=tk.LEFT)

        self.btn_export_csv = ttk.Button(frame_plot, text="Export Summary", command=self._export_csv, state=tk.DISABLED)
        self.btn_export_csv.pack(side=tk.LEFT, padx=(15, 0))

        # 因子負荷行列・因子得点の表示領域
        self.result_text = tk.Text(frame_bottom_right, wrap=tk.NONE, font=(get_japanese_monospace_font(), 11))
        scroll_y = ttk.Scrollbar(frame_bottom_right, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.df = _read_sd_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                self.df = _read_sd_csv(path, encoding="cp932")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
            return

        self.filtered_df = None

        self.file_path_var.set(path)

        thumbnail_folder = find_thumbnail_png_folder(path)
        if thumbnail_folder is not None:
            self.png_folder_var.set(str(thumbnail_folder))

        # カラム一覧を刺激名コンボボックスに設定
        columns = list(self.df.columns)
        self.stimulus_col_combo["values"] = columns
        if columns:
            self.stimulus_col_combo.current(0)

        # 回答者名カラム候補を設定（空欄 + 全カラム）
        self.resp_col_combo["values"] = [""] + columns
        self.resp_col_var.set("")

        # 数値カラムを形容詞対候補としてチェックボックス表示
        self._populate_checkboxes()

    def _populate_checkboxes(self):
        # 既存のチェックボックスをクリア
        for w in self.check_frame.winfo_children():
            w.destroy()
        self.check_vars.clear()

        if self.df is None:
            return

        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            var = tk.BooleanVar(value=True)
            self.check_vars[col] = var
            cb = ttk.Checkbutton(self.check_frame, text=col, variable=var, command=self._update_stats_tree)
            cb.pack(anchor=tk.W, pady=1)
            self._bind_adjective_mousewheel(cb)

        self.adjective_canvas.yview_moveto(0)
        self._update_stats_tree()

    def _update_adjective_scrollregion(self, _event=None):
        """形容詞対一覧のサイズ変更を縦スクロール範囲に反映する。"""
        bounds = self.adjective_canvas.bbox("all")
        if bounds is not None:
            self.adjective_canvas.configure(scrollregion=bounds)

    def _resize_adjective_check_frame(self, event):
        """キャンバス幅に合わせてチェックボックス領域を広げる。"""
        self.adjective_canvas.itemconfigure(self._adjective_window, width=event.width)

    def _bind_adjective_mousewheel(self, widget):
        """形容詞対一覧上でのみマウスホイールを有効にする。"""
        widget.bind("<MouseWheel>", self._scroll_adjective_pairs)
        widget.bind("<Button-4>", self._scroll_adjective_pairs)
        widget.bind("<Button-5>", self._scroll_adjective_pairs)

    def _scroll_adjective_pairs(self, event):
        """Windows・macOS・Linuxのホイール操作で形容詞対一覧を動かす。"""
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        self.adjective_canvas.yview_scroll(units, "units")
        return "break"

    def _format_adj_name(self, col):
        """正規表現で形容詞対カラム名を 'ADJ1 - ADJ2' 形式に変換する。因子負荷が負の場合は反転。"""
        pattern = self.regex_var.get().strip()
        if not pattern:
            return col
        try:
            m = re.search(pattern, col)
            if m and len(m.groups()) >= 2:
                adj1, adj2 = m.group(1), m.group(2)
                if self.invert_map.get(col, False):
                    return f"{adj2} - {adj1}"
                return f"{adj1} - {adj2}"
        except re.error:
            pass
        return col

    def _add_png_folder_control(self, parent):
        """メインウィンドウに共有PNGフォルダの選択コントロールを追加する。"""
        frame_folder = ttk.LabelFrame(parent, text="Thumbnail PNG Folder", padding=10)
        frame_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Entry(frame_folder, textvariable=self.png_folder_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(
            frame_folder,
            text="Select Folder...",
            command=self._select_png_folder,
        ).pack(side=tk.LEFT)

    def _select_png_folder(self):
        """PNGフォルダを選択し、アプリのセッション状態として保持する。"""
        current_folder = self.png_folder_var.get()
        initial_dir = current_folder if current_folder and os.path.isdir(current_folder) else os.path.expanduser("~")
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Select Folder Containing PNG Images",
            initialdir=initial_dir,
            mustexist=True,
        )
        if folder:
            self.png_folder_var.set(folder)

    def _create_png_preview_canvas(self, parent, initial_message):
        """刺激マップ・刺激フィルターで共用するPNGプレビュー領域を作成する。"""
        preview_canvas = tk.Canvas(parent, background="white", highlightthickness=0)
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._show_png_preview_message(preview_canvas, initial_message)
        return preview_canvas

    def _show_png_preview_message(self, preview_canvas, message):
        preview_canvas.delete("all")
        preview_canvas.update_idletasks()
        width = max(preview_canvas.winfo_width(), 200)
        height = max(preview_canvas.winfo_height(), 150)
        preview_canvas.create_text(
            width / 2,
            height / 2,
            text=message,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            width=max(width - 40, 160),
            fill="gray40",
        )
        preview_canvas.preview_photo = None

    def _show_png_preview(self, preview_canvas, dialog, path, stimulus_id):
        try:
            photo = tk.PhotoImage(master=dialog, file=str(path))
        except tk.TclError as error:
            self._show_png_preview_message(preview_canvas, f"Could not load {path.name}:\n{error}")
            return

        preview_canvas.update_idletasks()
        canvas_width = max(preview_canvas.winfo_width(), 200)
        canvas_height = max(preview_canvas.winfo_height(), 150)
        available_width = max(canvas_width - 20, 1)
        available_height = max(canvas_height - 50, 1)
        sample = max(
            1,
            (photo.width() + available_width - 1) // available_width,
            (photo.height() + available_height - 1) // available_height,
        )
        if sample > 1:
            photo = photo.subsample(sample, sample)

        preview_canvas.delete("all")
        preview_canvas.create_text(
            canvas_width / 2,
            10,
            text=f"Stimulus: {stimulus_id}   File: {path.name}",
            anchor=tk.N,
        )
        preview_canvas.create_image(
            canvas_width / 2,
            (canvas_height + 30) / 2,
            image=photo,
            anchor=tk.CENTER,
        )
        # Tk側で参照を保持し、画像がガベージコレクションされないようにする。
        preview_canvas.preview_photo = photo

    def _show_stimulus_png(self, preview_canvas, dialog, stimulus_id):
        """選択中の共有フォルダから刺激IDに対応するPNGを表示する。"""
        folder = self.png_folder_var.get()
        if not folder:
            self._show_png_preview_message(preview_canvas, "Select a thumbnail PNG folder in the main window first.")
            return

        png_path = find_stimulus_png(folder, stimulus_id)
        if png_path is None:
            self._show_png_preview_message(
                preview_canvas,
                f"PNG not found for stimulus: {stimulus_id}\n\n"
                "The filename must be <stimulus_id>.png, including any leading zeros.",
            )
            return
        self._show_png_preview(preview_canvas, dialog, png_path, stimulus_id)

    def _bind_stimulus_png_hover(self, widget, preview_canvas, dialog, stimulus_id):
        """刺激ラベルへのホバーで対応するPNGを表示する。"""
        widget.bind(
            "<Enter>",
            lambda _event: self._show_stimulus_png(preview_canvas, dialog, stimulus_id),
        )

    def _open_stimulus_filter_dialog(self):
        """分析対象の刺激を選択するダイアログ"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return
        stimulus_col = self.stimulus_col_var.get()
        if stimulus_col not in self.target_stimulus_table.keys():
            self.target_stimulus_table[stimulus_col] = sorted(self.df[stimulus_col].unique().tolist())
        all_stimuli = sorted(self.df[stimulus_col].unique().tolist())
        target_stimuli = self.target_stimulus_table.get(stimulus_col, all_stimuli)

        # ダイアログを作成
        dialog = tk.Toplevel(self.root)
        dialog.title("Filter Stimuli")
        dialog.transient(self.root)
        dialog.grab_set()
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        dialog.geometry(f"900x600+{x}+{y}")
        dialog.minsize(650, 400)

        paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        frame_stimuli = ttk.LabelFrame(paned, text="Stimuli", padding=5)
        frame_preview = ttk.LabelFrame(paned, text="PNG Preview", padding=5)
        paned.add(frame_stimuli, weight=2)
        paned.add(frame_preview, weight=3)

        stimulus_canvas = tk.Canvas(frame_stimuli, highlightthickness=0)
        stimulus_scrollbar = tk.Scrollbar(
            frame_stimuli,
            orient=tk.VERTICAL,
            command=stimulus_canvas.yview,
            width=16,
        )
        stimulus_list = ttk.Frame(stimulus_canvas)
        stimulus_window = stimulus_canvas.create_window((0, 0), window=stimulus_list, anchor="nw")
        stimulus_list.bind(
            "<Configure>",
            lambda _event: stimulus_canvas.configure(scrollregion=stimulus_canvas.bbox("all")),
        )
        stimulus_canvas.bind(
            "<Configure>",
            lambda event: stimulus_canvas.itemconfigure(stimulus_window, width=event.width),
        )
        stimulus_canvas.configure(yscrollcommand=stimulus_scrollbar.set)
        stimulus_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        stimulus_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        initial_message = (
            "Hover over a stimulus label to display its PNG."
            if self.png_folder_var.get()
            else "Select a thumbnail PNG folder in the main window, then hover over a stimulus label."
        )
        preview_canvas = self._create_png_preview_canvas(frame_preview, initial_message)

        # 刺激のチェックボックスを配置
        check_vars = {}
        for stimulus in all_stimuli:
            var = tk.BooleanVar(value=stimulus in target_stimuli)
            check_vars[stimulus] = var
            cb = ttk.Checkbutton(stimulus_list, text=stimulus, variable=var)
            cb.pack(anchor=tk.W, pady=1, padx=10)
            self._bind_stimulus_png_hover(cb, preview_canvas, dialog, stimulus)

        # OKボタン
        def on_ok():
            stimuli = [stimulus for stimulus, var in check_vars.items() if var.get()]
            self.target_stimulus_table[stimulus_col] = stimuli
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=(5, 10))

        # Tk側で参照を保持し、ダイアログ表示中にCanvasが破棄されないようにする。
        dialog.stimulus_canvas = stimulus_canvas
        dialog.preview_canvas = preview_canvas

    def _apply_regex(self):
        """正規表現を適用してTreeviewの表示名を更新する。"""
        self._update_stats_tree()

    def _update_stats_tree(self):
        """選択中の形容詞対カラムの平均・標準偏差（＋因子負荷量）をTreeviewに表示する。"""
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        # 因子負荷量がある場合はカラムを動的に追加
        if self.pattern_loading_df is not None:
            factor_cols = list(self.pattern_loading_df.columns)
            all_cols = ["mean", "std"] + factor_cols
        else:
            factor_cols = []
            all_cols = ["mean", "std"]

        self.stats_tree["columns"] = all_cols
        for c in all_cols:
            self.stats_tree.heading(c, text=c.capitalize() if c in ("mean", "std") else c, anchor=tk.CENTER)
            self.stats_tree.column(c, width=70, anchor=tk.CENTER, stretch=False)
        self.stats_tree.column("#0", width=180, stretch=True)
        self.stats_tree.heading("#0", text="Adjective Pair", anchor=tk.W)

        if self.df is None:
            return

        selected_cols = [col for col, var in self.check_vars.items() if var.get()]

        # pattern_loading_dfがある場合はそのソート順（best_factor, max_abs_loading）に従う
        if self.pattern_loading_df is not None:
            sorted_cols = [c for c in self.pattern_loading_df.index if c in selected_cols]
            # pattern_loading_dfに含まれない選択カラムは末尾に追加
            sorted_cols += [c for c in selected_cols if c not in sorted_cols]
        else:
            sorted_cols = selected_cols

        for col in sorted_cols:
            stats_df = self.filtered_df if self.filtered_df is not None else self.df
            inverted = self.invert_map.get(col, False)
            mean_val = stats_df[col].mean()
            std_val = stats_df[col].std()
            if inverted:
                mean_val = int(self.scale_var.get()) + 1 - mean_val
            row_vals = [f"{mean_val:.3f}", f"{std_val:.3f}"]
            if self.pattern_loading_df is not None and col in self.pattern_loading_df.index:
                for fc in factor_cols:
                    val = self.pattern_loading_df.at[col, fc]
                    if inverted:
                        val = -val
                    row_vals.append(f"{val:.3f}")
            else:
                row_vals.extend([""] * len(factor_cols))
            display_name = self._format_adj_name(col)
            self.stats_tree.insert("", tk.END, text=display_name, values=row_vals)

    def _on_corr_change(self, *args):
        # polychoricを選択したとき、シミュレーション回数をentryで指定できるようにする
        if self.corr_name_var.get() == "polychoric":
            self.parallel_analysis_iter_var.set("20")
            self.polychoric_entry.configure(state="normal")
        else:
            self.parallel_analysis_iter_var.set("500")
            self.polychoric_entry.configure(state="readonly")

    def _update_parallel_progress(self, current, total):
        if total <= 0:
            percent = 0
        else:
            percent = max(0, min(100, current / total * 100))
        self.parallel_progress_var.set(percent)
        self.parallel_progress_label_var.set(f"Parallel: {current} / {total}")
        self.root.update_idletasks()

    def _run_analysis(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return

        stimulus_col = self.stimulus_col_var.get()
        if not stimulus_col:
            messagebox.showwarning("Warning", "Please select a stimulus column.")
            return

        selected_cols = [col for col, var in self.check_vars.items() if var.get()]
        if not selected_cols:
            messagebox.showwarning("Warning", "Please select at least one adjective pair column.")
            return

        self.btn_run_analysis.config(state=tk.DISABLED)
        try:
            target_stimuli = self.target_stimulus_table.get(stimulus_col, self.df[stimulus_col].unique())
            filtered_df = self.df[self.df[stimulus_col].isin(target_stimuli)].copy()
            self.filtered_df = filtered_df

            if self.corr_name_var.get() == "polychoric":
                n_iter = int(self.parallel_analysis_iter_var.get())
            else:
                # Pearsonでは標準的なシミュレーション回数を使用
                n_iter = 500
            self._update_parallel_progress(0, n_iter)
            suggested_factors, parallel_str = print_parallel_analysis_summary(
                filtered_df,
                selected_cols,
                corr=self.corr_name_var.get(),
                n_iter=n_iter,
                progress_callback=self._update_parallel_progress,
            )
            self.parallel_progress_label_var.set(f"Parallel: {n_iter} / {n_iter}")
            self.root.update_idletasks()

            self.parallel_text.delete("1.0", tk.END)
            self.parallel_text.insert(tk.END, parallel_str)

            max_factors = len(selected_cols)
            numeric_max = min(max_factors, 10)

            selected_factor_mode = self.n_factors_var.get()
            if selected_factor_mode == "PA":
                n_factors = min(max(suggested_factors, 1), 10, max_factors)
            elif selected_factor_mode.isdigit():
                n_factors = min(max(int(selected_factor_mode), 1), numeric_max)
                self.n_factors_var.set(str(n_factors))
            else:
                self.n_factors_var.set("PA")
                n_factors = min(max(suggested_factors, 1), 10, max_factors)

            # 全回答者データで因子分析を実行
            factor_names = [f"Factor{i + 1}" for i in range(n_factors)]
            self.current_rotation = self.rotation_var.get()

            pattern_loading_df, structure_loading_df, factor_score_df, factor_corr_df = factor_analysis(
                filtered_df,
                selected_cols,
                factor_names,
                rotation=self.current_rotation,
                corr=self.corr_name_var.get(),
            )

            # 因子得点に識別用カラムを付与し、集計単位ごとに平均と標準偏差を計算
            resp_col = self.resp_col_var.get()
            if resp_col:
                factor_score_df[resp_col] = filtered_df.loc[factor_score_df.index, resp_col].values
            factor_score_df[stimulus_col] = filtered_df.loc[factor_score_df.index, stimulus_col].values
            group_cols = [resp_col, stimulus_col] if resp_col else [stimulus_col]
            score_df, score_summary_df = summarize_factor_scores(factor_score_df, group_cols, factor_names)
            # Sort factors
            pattern_loading_df["max_abs_loading"] = pattern_loading_df.abs().max(axis=1)
            pattern_loading_df["best_factor"] = pattern_loading_df.abs().idxmax(axis=1)
            pattern_loading_df = pattern_loading_df.sort_values(
                ["best_factor", "max_abs_loading"], ascending=[True, False]
            )

            # 因子負荷が負の形容詞対を反転させるためのマップを作成
            self.invert_map = {
                col: bool(pattern_loading_df.loc[col, pattern_loading_df.loc[col, "best_factor"]] < 0)
                for col in pattern_loading_df.index
            }

            sorted_index = pattern_loading_df.index
            pattern_loading_df = pattern_loading_df.drop(columns=["max_abs_loading", "best_factor"])
            if structure_loading_df is not None:
                structure_loading_df = structure_loading_df.loc[sorted_index]

            # 結果を表示
            self.result_text.delete("1.0", tk.END)

            self.result_text.insert(tk.END, score_summary_df.round(3).to_string() + "\n")

            self.pattern_loading_df = pattern_loading_df
            self.structure_loading_df = structure_loading_df
            # 刺激マップでは平均因子得点のみを使用する
            self.score_df = score_df
            self.score_summary_df = score_summary_df
            self.factor_corr_df = factor_corr_df
            self.factor_names = factor_names
            self.btn_apply_reg.config(state=tk.NORMAL)
            self.btn_plot_loadings.config(state=tk.NORMAL)
            self.btn_export_loadings.config(state=tk.NORMAL)
            self.btn_plot_map.config(state=tk.NORMAL if len(factor_names) >= 2 else tk.DISABLED)
            self.btn_export_csv.config(state=tk.NORMAL)

            self._update_stats_tree()

        except Exception as e:
            self.parallel_progress_label_var.set("Analysis: failed")
            messagebox.showerror("Error", f"Factor analysis failed:\n{e}")
        finally:
            self.btn_run_analysis.config(state=tk.NORMAL)

    def _plot_loadings(self):
        if self.pattern_loading_df is not None:
            if self.current_rotation == "varimax":
                rotation_label = "Varimax Rotation"
            elif self.current_rotation == "promax":
                rotation_label = "Promax Rotation"
            else:
                rotation_label = "No Rotation"
            matrix_label = "Factor Pattern Matrix" if self.current_rotation == "promax" else "Factor Loading Matrix"
            title = f"{matrix_label} ({rotation_label})"
            # 反転を反映したコピーを作成
            plot_df = self.pattern_loading_df.copy()
            original_cols = list(plot_df.index)
            for col in original_cols:
                if self.invert_map.get(col, False):
                    plot_df.loc[col] = -plot_df.loc[col]
            # 表示名に変換
            plot_df.index = [self._format_adj_name(col) for col in original_cols]
            inverted_rows = [self.invert_map.get(col, False) for col in original_cols]
            if self.corr_name_var.get() == "polychoric":
                caption = "Corr: Polychoric\n"
                caption += f"Parallel analysis iterations: {self.parallel_analysis_iter_var.get()}  Percentile: 95th\n"
            else:
                caption = "Corr: Pearson\n"
                caption += "Parallel analysis percentile: 95th\n"

            # 各因子に対してCronbach's alphaを計算してキャプションに追加
            factor_items = pickup_by_factor(self.pattern_loading_df)
            alpha_df = self.filtered_df if self.filtered_df is not None else self.df
            scale_num = int(self.scale_var.get())
            cronbach_caption = f"Cronbach's alpha (filtered n={len(alpha_df)}):\n"
            for factor in factor_items:
                if len(factor["items"]) >= 2:
                    alpha = compute_cronbach_alpha(
                        alpha_df[factor["items"]], factor["items"], factor["invert"], 1.0, scale_num
                    )
                    cronbach_caption += f"{factor['factor_name']}: {alpha:.3f},  "
                else:
                    cronbach_caption += f"{factor['factor_name']}: N/A (only {len(factor['items'])} item),  "

            plot_factor_loadings(
                plot_df,
                title=title,
                inverted_rows=inverted_rows,
                promax_corr_df=self.factor_corr_df if self.current_rotation == "promax" else None,
                caption=caption + cronbach_caption,
            )

    def _export_loadings_csv(self):
        if self.pattern_loading_df is None:
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = filedialog.asksaveasfilename(
            title="Export Factor Loadings",
            initialdir=desktop,
            initialfile="factor_loadings.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        export_df = self.pattern_loading_df.copy()
        # _plot_loadings と同じ手順で反転と表示名変換を適用
        original_cols = list(export_df.index)
        for col in original_cols:
            if self.invert_map.get(col, False):
                export_df.loc[col] = -export_df.loc[col]
        export_df.index = [self._format_adj_name(col) for col in original_cols]
        export_df.index.name = "Adjective Pair"
        export_df.round(3).to_csv(path, encoding="utf-8-sig")

        messagebox.showinfo("Export", f"Saved to:\n{path}")

    def _export_csv(self):
        if self.score_summary_df is None:
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = filedialog.asksaveasfilename(
            title="Export Factor Scores",
            initialdir=desktop,
            initialfile="factor_scores.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        self.score_summary_df.round(3).to_csv(path, encoding="utf-8-sig")
        messagebox.showinfo("Export", f"Saved to:\n{path}")

    def _plot_stimulus_map(self):
        stimulus_level = self.stimulus_col_var.get() if self.resp_col_var.get() else None
        dialog = tk.Toplevel(self.root)
        dialog.title("Stimulus Map")
        dialog.geometry("1200x740")
        dialog.minsize(800, 540)
        dialog.transient(self.root)

        # PCA・Factor軸で共用する表示コントロール
        frame_controls = ttk.Frame(dialog, padding=(10, 8, 10, 0))
        frame_controls.pack(fill=tk.X)
        plot_mode_var = tk.StringVar(master=dialog, value="pca")
        x_factor_var = tk.StringVar(master=dialog, value=self.factor_names[0])
        y_factor_var = tk.StringVar(master=dialog, value=self.factor_names[1])

        # 左に刺激マップ、右に画像プレビューを表示する領域を配置
        paned = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        frame_map = ttk.LabelFrame(paned, text="PCA Map", padding=5)
        frame_preview = ttk.LabelFrame(paned, text="PNG Preview", padding=5)
        paned.add(frame_map, weight=3)
        paned.add(frame_preview, weight=2)

        initial_message = (
            "Click a stimulus point to display its PNG."
            if self.png_folder_var.get()
            else "Select a thumbnail PNG folder in the main window, then click a stimulus point."
        )
        preview_canvas = self._create_png_preview_canvas(frame_preview, initial_message)

        def render_map():
            if plot_mode_var.get() == "factors":
                x_factor = x_factor_var.get()
                y_factor = y_factor_var.get()
                if x_factor == y_factor:
                    messagebox.showwarning(
                        "Stimulus Map",
                        "Select two different factors for the X and Y axes.",
                        parent=dialog,
                    )
                    return
                fig = create_factor_map_figure(
                    self.score_df,
                    x_factor,
                    y_factor,
                    title=f"Stimulus Map ({x_factor} × {y_factor})",
                    stimulus_level=stimulus_level,
                )
                frame_map.configure(text="Factor Axes Map")
            else:
                fig = create_pca_map_figure(
                    self.score_df,
                    self.factor_names,
                    title="Stimulus Map (2D PCA with Factor Axes)",
                    stimulus_level=stimulus_level,
                )
                frame_map.configure(text="PCA Map")

            for child in frame_map.winfo_children():
                child.destroy()

            map_canvas = FigureCanvasTkAgg(fig, master=frame_map)
            map_canvas.draw()
            map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            def on_pick(event):
                stimulus_ids = fig.stimulus_pick_targets.get(event.artist)
                if not stimulus_ids:
                    return
                stimulus_id = stimulus_ids[int(event.ind[0])]
                self._show_stimulus_png(preview_canvas, dialog, stimulus_id)

            pick_connection_id = map_canvas.mpl_connect("pick_event", on_pick)

            # Tk側で参照を保持し、ダイアログ表示中にCanvasが破棄されないようにする
            dialog.map_figure = fig
            dialog.map_canvas = map_canvas
            dialog.pick_connection_id = pick_connection_id

        def on_plot_mode_change():
            factor_state = "readonly" if plot_mode_var.get() == "factors" else tk.DISABLED
            x_factor_combo.configure(state=factor_state)
            y_factor_combo.configure(state=factor_state)
            render_map()

        ttk.Label(frame_controls, text="Plot:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(
            frame_controls,
            text="PCA",
            variable=plot_mode_var,
            value="pca",
            command=on_plot_mode_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            frame_controls,
            text="Factor Axes",
            variable=plot_mode_var,
            value="factors",
            command=on_plot_mode_change,
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(frame_controls, text="X:").pack(side=tk.LEFT, padx=(0, 2))
        x_factor_combo = ttk.Combobox(
            frame_controls,
            textvariable=x_factor_var,
            values=self.factor_names,
            state=tk.DISABLED,
            width=12,
        )
        x_factor_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(frame_controls, text="Y:").pack(side=tk.LEFT, padx=(0, 2))
        y_factor_combo = ttk.Combobox(
            frame_controls,
            textvariable=y_factor_var,
            values=self.factor_names,
            state=tk.DISABLED,
            width=12,
        )
        y_factor_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(frame_controls, text="Update Map", command=render_map).pack(side=tk.LEFT)

        dialog.plot_mode_var = plot_mode_var
        dialog.x_factor_var = x_factor_var
        dialog.y_factor_var = y_factor_var
        dialog.preview_canvas = preview_canvas
        render_map()


def main():
    root = tk.Tk()
    SDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
