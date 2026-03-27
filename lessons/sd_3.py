import pandas as pd
import sd_utils

# 日本語表示をいい感じにする処理を呼び出す
sd_utils.set_japanese_font()

# データが入っているsample_sd.csvのパスをcsv_file_pathに格納
csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)

# 集計したい対象物コードを含むデータだけを抽出してtar_dfに格納
target_obj = "o_001"
tar_df = src_df[src_df["対象物コード"] == target_obj].copy()

tar_cols = [c for c in tar_df.columns if c.startswith("評価.")]

# 集計表を作成(lickert_3.py参照)
melted_df = tar_df.melt(value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="value", aggfunc="size", fill_value=0)
heatmap_df = heatmap_df.reindex(index=tar_cols, columns=range(1, 8), fill_value=0)

print(f"評定の件数 ({target_obj}):")
print(heatmap_df)
