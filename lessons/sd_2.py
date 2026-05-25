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
print(f"ファイル名: {csv_file_path}\n行数: {num_rows}行")

# "stimulus_id"カラムのユニークな値をstimulus_namesに格納して表示
stimulus_names = src_df["stimulus_id"].unique()
print("\n'stimulus_id' の一覧:")
print(stimulus_names)

# "respondent_id"カラムのユニークな値をsbj_namesに格納して表示
sbj_names = src_df["respondent_id"].unique()
print("\n'respondent_id' の一覧:")
print(sbj_names)
