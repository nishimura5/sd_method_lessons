import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from sd_plot import plot_factor_loadings, plot_pca
from sd_utils import factor_analysis, get_japanese_monospace_font, set_japanese_font
from tooltip import ToolTip


class SDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SD Method Factor Analysis Tool")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        self.df = None
        self.check_vars = {}
        self.loading_df = None
        self.score_df = None
        self.factor_names = None
        self.tar_obj_table = {}  # 分析対象の刺激名のホワイトリスト、{"colname": [obj1, obj2, ...], ...} の形式
        self.invert_map = {}

        set_japanese_font()
        self._build_ui()

    def _build_ui(self):
        # === CSVファイル選択 ===
        frame_file = ttk.LabelFrame(self.root, text="Select CSV File", padding=10)
        frame_file.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.file_path_var = tk.StringVar()
        ttk.Entry(frame_file, textvariable=self.file_path_var, state="readonly", width=70).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(frame_file, text="Browse", command=self._select_file).pack(side=tk.LEFT)

        # === 刺激名カラム選択 ===
        frame_row = ttk.Frame(self.root)
        frame_row.pack(fill=tk.X, padx=10, pady=5)

        frame_obj = ttk.LabelFrame(frame_row, text="Stimulus Column", padding=10)
        frame_obj.pack(side=tk.LEFT, fill=tk.X, padx=(0, 5))

        self.obj_col_var = tk.StringVar()
        self.obj_col_combo = ttk.Combobox(frame_obj, textvariable=self.obj_col_var, state="readonly", width=16)
        self.obj_col_combo.pack(side=tk.LEFT)

        # === 刺激名フィルターダイアログを開くボタン ===
        ttk.Button(frame_obj, text="Filter...", command=self._open_stimulus_filter_dialog).pack(
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
            '[Example]\n  original: "Q1_warm_cold"\n  regex: Q\\d+_(.+)_(.+)\n  result: warm - cold',
        )
        ttk.Button(frame_regex, text="Apply", command=self._apply_regex).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame_regex, text="Scale:").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="7")
        ttk.Combobox(frame_regex, textvariable=self.scale_var, state="readonly", values=["5", "7"], width=3).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # === 左右分割: 形容詞対カラム選択（左）と結果表示（右） ===
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        # --- 左側: 形容詞対カラム選択 ---
        frame_adj = ttk.LabelFrame(paned, text="Select Adjective Pair Columns", padding=10)
        paned.add(frame_adj, weight=1)

        # スクロール可能なチェックボックス領域
        canvas = tk.Canvas(frame_adj, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame_adj, orient=tk.VERTICAL, command=canvas.yview)
        self.check_frame = ttk.Frame(canvas)

        self.check_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.check_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # マウスホイールでスクロール
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120 or (1 if event.delta > 0 else -1)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- 中央: treeviewで各形容詞対のmeanとstdを表示 ---
        frame_center = ttk.LabelFrame(paned, text="Adjective Pair Statistics", padding=10)
        paned.add(frame_center, weight=1)

        # 因子数選択と実行
        frame_exec = ttk.Frame(frame_center)
        frame_exec.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(frame_exec, text="Factors:").pack(side=tk.LEFT)
        self.n_factors_var = tk.StringVar(value="3")
        factor_combo = ttk.Combobox(
            frame_exec, textvariable=self.n_factors_var, state="readonly", values=["2", "3", "4", "5"], width=5
        )
        factor_combo.pack(side=tk.LEFT, padx=(5, 15))

        self.varimax_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_exec, text="Varimax", variable=self.varimax_var).pack(side=tk.LEFT, padx=(0, 15))
        self.use_varimax = self.varimax_var.get()

        ttk.Button(frame_exec, text="Run Analysis", command=self._run_analysis).pack(side=tk.LEFT)

        self.btn_plot_loadings = ttk.Button(
            frame_exec, text="Plot Loadings", command=self._plot_loadings, state=tk.DISABLED
        )
        self.btn_plot_loadings.pack(side=tk.LEFT, padx=(15, 5))
        self.btn_export_loadings = ttk.Button(
            frame_exec, text="Export Loadings", command=self._export_loadings_csv, state=tk.DISABLED
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
        frame_bottom = ttk.LabelFrame(self.root, text="Factor Scores by Stimulus", padding=10)
        frame_bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # グラフ描画ボタン
        frame_plot = ttk.Frame(frame_bottom)
        frame_plot.pack(fill=tk.X, pady=(0, 5))

        self.btn_plot_pca = ttk.Button(frame_plot, text="Plot PCA", command=self._plot_pca, state=tk.DISABLED)
        self.btn_plot_pca.pack(side=tk.LEFT)

        self.btn_export_csv = ttk.Button(frame_plot, text="Export CSV", command=self._export_csv, state=tk.DISABLED)
        self.btn_export_csv.pack(side=tk.LEFT, padx=(15, 0))

        # 因子負荷行列・因子得点の表示領域
        self.result_text = tk.Text(frame_bottom, wrap=tk.NONE, font=(get_japanese_monospace_font(), 11))
        scroll_y = ttk.Scrollbar(frame_bottom, orient=tk.VERTICAL, command=self.result_text.yview)
        scroll_x = ttk.Scrollbar(frame_bottom, orient=tk.HORIZONTAL, command=self.result_text.xview)
        self.result_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(path, encoding="cp932")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
            return

        self.file_path_var.set(path)

        # カラム一覧を刺激名コンボボックスに設定
        columns = list(self.df.columns)
        self.obj_col_combo["values"] = columns
        if columns:
            self.obj_col_combo.current(0)

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

        self._update_stats_tree()

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

    def _open_stimulus_filter_dialog(self):
        """分析対象の刺激を選択するダイアログ"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return
        obj_col = self.obj_col_var.get()
        if obj_col not in self.tar_obj_table.keys():
            self.tar_obj_table[obj_col] = sorted(self.df[obj_col].unique().tolist())
        all_obj_list = sorted(self.df[obj_col].unique().tolist())
        tar_obj_list = self.tar_obj_table.get(obj_col, all_obj_list)

        # ダイアログを作成
        dialog = tk.Toplevel(self.root)
        dialog.title("Filter Stimuli")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        # 刺激のチェックボックスを配置
        check_vars = {}
        for obj in all_obj_list:
            var = tk.BooleanVar(value=obj in tar_obj_list)
            check_vars[obj] = var
            cb = ttk.Checkbutton(frame, text=obj, variable=var)
            cb.pack(anchor=tk.W, pady=1, padx=10)

        # OKボタン
        def on_ok():
            obj_list = [obj for obj, var in check_vars.items() if var.get()]
            self.tar_obj_table[obj_col] = obj_list
            dialog.destroy()

        ttk.Button(frame, text="OK", command=on_ok).pack(pady=10)

    def _apply_regex(self):
        """正規表現を適用してTreeviewの表示名を更新する。"""
        self._update_stats_tree()

    def _update_stats_tree(self):
        """選択中の形容詞対カラムの平均・標準偏差（＋因子負荷量）をTreeviewに表示する。"""
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        # 因子負荷量がある場合はカラムを動的に追加
        if self.loading_df is not None:
            factor_cols = list(self.loading_df.columns)
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

        # loading_dfがある場合はそのソート順（best_factor, max_abs_loading）に従う
        if self.loading_df is not None:
            sorted_cols = [c for c in self.loading_df.index if c in selected_cols]
            # loading_dfに含まれない選択カラムは末尾に追加
            sorted_cols += [c for c in selected_cols if c not in sorted_cols]
        else:
            sorted_cols = selected_cols

        for col in sorted_cols:
            inverted = self.invert_map.get(col, False)
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            if inverted:
                mean_val = int(self.scale_var.get()) + 1 - mean_val
            row_vals = [f"{mean_val:.3f}", f"{std_val:.3f}"]
            if self.loading_df is not None and col in self.loading_df.index:
                for fc in factor_cols:
                    val = self.loading_df.at[col, fc]
                    if inverted:
                        val = -val
                    row_vals.append(f"{val:.3f}")
            else:
                row_vals.extend([""] * len(factor_cols))
            display_name = self._format_adj_name(col)
            self.stats_tree.insert("", tk.END, text=display_name, values=row_vals)

    def _run_analysis(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return

        obj_col = self.obj_col_var.get()
        if not obj_col:
            messagebox.showwarning("Warning", "Please select a stimulus column.")
            return

        selected_cols = [col for col, var in self.check_vars.items() if var.get()]
        if not selected_cols:
            messagebox.showwarning("Warning", "Please select at least one adjective pair column.")
            return

        n_factors = int(self.n_factors_var.get())

        if n_factors > len(selected_cols):
            messagebox.showwarning(
                "Warning",
                f"Number of factors ({n_factors}) exceeds the number of selected columns ({len(selected_cols)}).",
            )
            return

        try:
            # 全回答者データで因子分析を実行
            factor_names = [f"Factor{i + 1}" for i in range(n_factors)]
            self.use_varimax = self.varimax_var.get()
            tar_stims = self.tar_obj_table.get(obj_col, self.df[obj_col].unique())
            filtered_df = self.df[self.df[obj_col].isin(tar_stims)]
            loading_df, factor_score_df = factor_analysis(
                filtered_df, selected_cols, factor_names, varimax=self.use_varimax
            )

            # 因子得点にオブジェクトカラムを付与し、オブジェクトごとに平均
            resp_col = self.resp_col_var.get()
            if resp_col:
                factor_score_df[resp_col] = filtered_df[resp_col].values
            factor_score_df[obj_col] = filtered_df[obj_col].values
            group_cols = [resp_col, obj_col] if resp_col else [obj_col]
            score_df = factor_score_df.groupby(group_cols).mean()
            # Sort factors
            loading_df["max_abs_loading"] = loading_df.abs().max(axis=1)
            loading_df["best_factor"] = loading_df.abs().idxmax(axis=1)
            loading_df = loading_df.sort_values(["best_factor", "max_abs_loading"], ascending=[True, False])

            # 因子負荷が負の形容詞対を反転させるためのマップを作成
            self.invert_map = {
                col: bool(loading_df.loc[col, loading_df.loc[col, "best_factor"]] < 0) for col in loading_df.index
            }

            loading_df = loading_df.drop(columns=["max_abs_loading", "best_factor"])

            # 結果を表示
            self.result_text.delete("1.0", tk.END)

            self.result_text.insert(tk.END, score_df.round(3).to_string() + "\n")

            self.loading_df = loading_df
            self.score_df = score_df
            self.factor_names = factor_names
            self.btn_plot_loadings.config(state=tk.NORMAL)
            self.btn_export_loadings.config(state=tk.NORMAL)
            self.btn_plot_pca.config(state=tk.NORMAL)
            self.btn_export_csv.config(state=tk.NORMAL)

            self._update_stats_tree()

        except Exception as e:
            messagebox.showerror("Error", f"Factor analysis failed:\n{e}")

    def _plot_loadings(self):
        if self.loading_df is not None:
            rotation_label = "Varimax Rotation" if self.use_varimax else "No Rotation"
            title = f"Factor Loading Matrix ({rotation_label})"
            # 反転を反映したコピーを作成
            plot_df = self.loading_df.copy()
            original_cols = list(plot_df.index)
            for col in original_cols:
                if self.invert_map.get(col, False):
                    plot_df.loc[col] = -plot_df.loc[col]
            # 表示名に変換
            plot_df.index = [self._format_adj_name(col) for col in original_cols]
            inverted_rows = [self.invert_map.get(col, False) for col in original_cols]
            plot_factor_loadings(plot_df, self.factor_names, title=title, inverted_rows=inverted_rows)

    def _export_loadings_csv(self):
        if self.loading_df is None:
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
        export_df = self.loading_df.copy()
        # _plot_loadings と同じ手順で反転と表示名変換を適用
        original_cols = list(export_df.index)
        for col in original_cols:
            if self.invert_map.get(col, False):
                export_df.loc[col] = -export_df.loc[col]
        export_df.index = [self._format_adj_name(col) for col in original_cols]
        export_df.index.name = "Adjective pair"
        export_df.round(3).to_csv(path, encoding="utf-8-sig")

        messagebox.showinfo("Export", f"Saved to:\n{path}")

    def _export_csv(self):
        if self.score_df is None:
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
        self.score_df.round(3).to_csv(path, encoding="utf-8-sig")
        messagebox.showinfo("Export", f"Saved to:\n{path}")

    def _plot_pca(self):
        if self.score_df is not None:
            plot_pca(self.score_df, self.factor_names, title="Stimulus Positions (PCA 2D + Factor Axis Annotation)")


def main():
    root = tk.Tk()
    SDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
