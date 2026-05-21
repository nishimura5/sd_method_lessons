import pandas as pd

import sd_utils

# 日本語表示をいい感じにする処理を呼び出す
sd_utils.set_japanese_font()

# データが入っているsample_sd.csvのパスをcsv_file_pathに格納
csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)

scale_cols = [col for col in src_df.columns if "-" in col]

# 全objの回答件数のsumを集計
melted_df = src_df.melt(value_vars=scale_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="value", aggfunc="size", fill_value=0)
heatmap_df = heatmap_df.reindex(index=scale_cols, columns=range(1, 8), fill_value=0)

# 標準偏差を計算
std_series = src_df[scale_cols].std()

# heat_map_dfの右に標準偏差の列を追加
heatmap_df["標準偏差"] = std_series

# 結果を表示
print("評定の件数 (全obj合計)と標準偏差:")
print(heatmap_df)
