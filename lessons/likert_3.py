import pandas as pd
import sd_utils

# 日本語表示をいい感じにする処理を呼び出す
sd_utils.set_japanese_font()

# データが入っているsample_1.csvのパスをcsv_file_pathに格納、自作のcsv_pathモジュールを使用
csv_file_path = sd_utils.get_csv_path("sample_1.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)

# 食べ物系の質問項目の選択肢は"ラーメン", "カレー"など共通しているので、まとめて集計する
# 集計対象の質問項目名をtar_colsに格納
tar_cols = ["ラーメンが好き", "寿司が好き", "カレーが好き", "焼肉が好き", "天丼が好き"]
one_seven = [7, 6, 5, 4, 3, 2, 1]

# src_dfのうち、集計対象の質問項目(tar_cols)を指定してmelt()し、縦長のデータに変換してmelted_dfに格納
# "value"カラムに選択肢の値が格納される
melted_df = src_df.melt(value_vars=tar_cols, var_name="質問項目")
# pivot_table()でaggfunc="size"を指定すると、同じ組み合わせのデータをカウントしてDataFrameにしてくれる
heatmap_df = melted_df.pivot_table(index="質問項目", columns="value", aggfunc="size", fill_value=0)
heatmap_df = heatmap_df.reindex(index=tar_cols, columns=one_seven, fill_value=0)

print("[likert_p3] Count table pivoted (questionnaire items x choices):")
print(heatmap_df)
