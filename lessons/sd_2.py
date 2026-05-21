import pandas as pd
import sd_utils

# 日本語表示をいい感じにする処理を呼び出す
sd_utils.set_japanese_font()

# データが入っているsample_sd.csvのパスをcsv_file_pathに格納、自作のcsv_pathモジュールを使用
csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)

# 行数をカウントする
num_rows = src_df.index.size
print(f"ファイル名: {csv_file_path}\n行数: {num_rows}行\n")

# "評価."で始まるカラムをscale_colsに格納
scale_cols = [col for col in src_df.columns if "-" in col]
print("\n形容詞対:")
for col in scale_cols:
    print(f"  {col}")

# "stimulus_id"カラムのユニークな値をobj_namesに格納して表示
obj_col_name = "stimulus_id"
obj_names = src_df[obj_col_name].unique()
print(f"\n'{obj_col_name}' の一覧:")
print(", ".join(obj_names))

# "respondent_id"カラムのユニークな値をsbj_namesに格納して表示
sbj_col_name = "respondent_id"
sbj_names = src_df[sbj_col_name].unique()
print(f"\n'{sbj_col_name}' の一覧:")
print(", ".join(sbj_names))
