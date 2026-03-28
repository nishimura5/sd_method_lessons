import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from sd_plot import plot_factor_loadings, plot_pca
from sd_utils import factor_analysis, get_japanese_monospace_font, set_japanese_font


class SDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SD Method Factor Analysis Tool")
        self.root.geometry("1200x700")
        self.root.minsize(800, 600)

        self.df = None
        self.check_vars = {}
        self.loading_df = None
        self.score_df = None
        self.factor_names = None

        set_japanese_font()
        self._build_ui()

    def _build_ui(self):
        # === CSVファイル選択 ===
        frame_file = ttk.LabelFrame(self.root, text="1. Select CSV File", padding=10)
        frame_file.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.file_path_var = tk.StringVar()
        ttk.Entry(frame_file, textvariable=self.file_path_var, state="readonly", width=70).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(frame_file, text="Browse", command=self._select_file).pack(side=tk.LEFT)

        # === 刺激名カラム選択 ===
        frame_obj = ttk.LabelFrame(self.root, text="2. Select Stimulus Column", padding=10)
        frame_obj.pack(fill=tk.X, padx=10, pady=5)

        self.obj_col_var = tk.StringVar()
        self.obj_col_combo = ttk.Combobox(frame_obj, textvariable=self.obj_col_var, state="readonly", width=40)
        self.obj_col_combo.pack(side=tk.LEFT)

        # === 左右分割: 形容詞対カラム選択（左）と結果表示（右） ===
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- 左側: 形容詞対カラム選択 ---
        frame_adj = ttk.LabelFrame(paned, text="3. Select Adjective Pair Columns", padding=10)
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

        # --- 右側: 結果表示 ---
        frame_right = ttk.Frame(paned)
        paned.add(frame_right, weight=3)

        # 因子数選択と実行
        frame_exec = ttk.LabelFrame(frame_right, text="4. Run Factor Analysis", padding=10)
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

        self.btn_plot_pca = ttk.Button(frame_exec, text="Plot PCA", command=self._plot_pca, state=tk.DISABLED)
        self.btn_plot_pca.pack(side=tk.LEFT)

        # 因子負荷行列・因子得点の表示領域
        frame_result = ttk.LabelFrame(frame_right, text="Factor Loading Matrix / Factor Scores", padding=10)
        frame_result.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(frame_result, wrap=tk.NONE, font=(get_japanese_monospace_font(), 11))
        scroll_y = ttk.Scrollbar(frame_result, orient=tk.VERTICAL, command=self.result_text.yview)
        scroll_x = ttk.Scrollbar(frame_result, orient=tk.HORIZONTAL, command=self.result_text.xview)
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
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
            return

        self.file_path_var.set(path)

        # カラム一覧を刺激名コンボボックスに設定
        columns = list(self.df.columns)
        self.obj_col_combo["values"] = columns
        if columns:
            self.obj_col_combo.current(0)

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
            ttk.Checkbutton(self.check_frame, text=col, variable=var).pack(anchor=tk.W, pady=1)

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
            loading_df, factor_score_df = factor_analysis(self.df, selected_cols, factor_names, varimax=self.use_varimax)

            # 因子得点にオブジェクトカラムを付与し、オブジェクトごとに平均
            factor_score_df[obj_col] = self.df[obj_col].values
            score_df = factor_score_df.groupby(obj_col).mean()
            # Sort factors
            loading_df["max_abs_loading"] = loading_df.abs().max(axis=1)
            loading_df["best_factor"] = loading_df.abs().idxmax(axis=1)
            loading_df = loading_df.sort_values(["best_factor", "max_abs_loading"], ascending=[True, False])
            loading_df = loading_df.drop(columns=["max_abs_loading", "best_factor"])

            # 結果を表示
            self.result_text.delete("1.0", tk.END)

            rotation_label = "Varimax Rotation" if self.use_varimax else "No Rotation"

            self.result_text.insert(tk.END, "=" * 60 + "\n")
            self.result_text.insert(tk.END, f" Factor Loading Matrix ({rotation_label})\n")
            self.result_text.insert(tk.END, "=" * 60 + "\n")
            self.result_text.insert(tk.END, loading_df.round(3).to_string() + "\n\n")

            self.result_text.insert(tk.END, "=" * 60 + "\n")
            self.result_text.insert(tk.END, " Factor Scores by Stimulus\n")
            self.result_text.insert(tk.END, "=" * 60 + "\n")
            self.result_text.insert(tk.END, score_df.round(3).to_string() + "\n")

            self.loading_df = loading_df
            self.score_df = score_df
            self.factor_names = factor_names
            self.btn_plot_loadings.config(state=tk.NORMAL)
            self.btn_plot_pca.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Factor analysis failed:\n{e}")

    def _plot_loadings(self):
        if self.loading_df is not None:
            rotation_label = "Varimax Rotation" if self.use_varimax else "No Rotation"
            title = f"Factor Loading Matrix ({rotation_label})"
            plot_factor_loadings(self.loading_df, self.factor_names, title=title)

    def _plot_pca(self):
        if self.score_df is not None:
            plot_pca(self.score_df, self.factor_names, title="Stimulus Positions (PCA 2D + Factor Axis Annotation)")


def main():
    root = tk.Tk()
    SDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
