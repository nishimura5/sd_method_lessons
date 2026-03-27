import pandas as pd
import sd_utils

# 日本語表示をいい感じにする処理を呼び出す
sd_utils.set_japanese_font()

# データが入っているsample_1.csvのパスをcsv_file_pathに格納、自作のcsv_pathモジュールを使用
csv_file_path = sd_utils.get_csv_path("sample_1.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)

# "寿司が好き"カラムを集計
hobby_count_sr = src_df["寿司が好き"].value_counts()
print("[likert_p2] Count of people by sushi preference:")
print(hobby_count_sr)

# 7段階分のindexをreindexで設定すると、並び順もきれいになり、ゼロ人の項目も表示される
one_seven = [7, 6, 5, 4, 3, 2, 1]
hobby_count_sr = src_df["寿司が好き"].value_counts().reindex(index=one_seven, fill_value=0)
print("\n[likert_p2] Count of people by sushi preference:")
print(hobby_count_sr)
