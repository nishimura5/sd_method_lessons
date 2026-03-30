import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# Number of factors to extract
# Note: The choice of n_factors can be guided by eigenvalue analysis
factor_names = ["因子1", "因子2", "因子3"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, tar_cols, factor_names)

factor_score_df["回答者コード"] = src_df["回答者コード"].values
factor_score_df["対象物コード"] = src_df["対象物コード"].values
factor_score_df = factor_score_df.set_index(["回答者コード", "対象物コード"])

# Mean factor scores by object (representative positions)
object_factor_df = factor_score_df.groupby("対象物コード").mean()
print(f"対象物ごとの{len(factor_names)}因子得点平均:")
print(object_factor_df.round(3))

print(f"\n回答者ごとの{len(factor_names)}因子得点:")
print(factor_score_df.round(3))
