import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 因子の名称を定義
factor_names = ["因子1", "因子2", "因子3"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, tar_cols, factor_names)

# Sort factors
rotated_loading_df["max_abs_loading"] = rotated_loading_df.abs().max(axis=1)
rotated_loading_df["best_factor"] = rotated_loading_df.abs().idxmax(axis=1)
rotated_loading_df = rotated_loading_df.sort_values(["best_factor", "max_abs_loading"], ascending=[True, False])
rotated_loading_df = rotated_loading_df.drop(columns=["max_abs_loading", "best_factor"])

print(rotated_loading_df.round(3))

# DesktopにCSV出力（エクセルで開く際の文字化け対策としてUTF-8 BOM付きでデスクトップに保存）
# rotated_loading_df.to_csv(sd_utils.set_csv_path("factor_loadings.csv"), encoding="utf-8-sig")
