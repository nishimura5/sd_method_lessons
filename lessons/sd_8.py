import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

scale_cols = [col for col in src_df.columns if "-" in col]

# Number of factors to extract
# Note: The choice of n_factors can be guided by eigenvalue analysis
factor_names = ["因子1", "因子2"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, scale_cols, factor_names)

factor_score_df["respondent_id"] = src_df["respondent_id"].values
factor_score_df["stimulus_id"] = src_df["stimulus_id"].values
factor_score_df = factor_score_df.set_index(["respondent_id", "stimulus_id"])

# Mean factor scores by stimulus (representative positions)
stimulus_factor_df = factor_score_df.groupby("stimulus_id").mean()
print(f"刺激ごとの{len(factor_names)}因子得点平均:")
print(stimulus_factor_df.round(3))

print(f"\n回答者ごとの{len(factor_names)}因子得点:")
print(factor_score_df.round(3))
