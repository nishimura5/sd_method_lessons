import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

scale_cols = [col for col in src_df.columns if "-" in col]

# 集計表を作成
melted_df = src_df.melt(id_vars=["stimulus_id"], value_vars=scale_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="stimulus_id", values="value", aggfunc="mean")

print("評定の平均値 (全刺激):")
print(heatmap_df)
