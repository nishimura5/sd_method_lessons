import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 集計表を作成
melted_df = src_df.melt(id_vars=["対象物コード"], value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="対象物コード", values="value", aggfunc="mean")
heatmap_df = heatmap_df.reindex(index=tar_cols)

print("評定の平均値 (全対象物):")
print(heatmap_df)
