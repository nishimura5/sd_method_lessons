import os

import pandas as pd

# このファイルが置かれているディレクトリをcurrent_dirに格納
current_dir = os.path.dirname(os.path.abspath(__file__))
# データが入っているsample_1.csvのパスをcsv_file_pathに格納
csv_file_path = os.path.join(current_dir, "..", "sample_data", "sample_1.csv")
# CSVファイルを読み込んでDataFrameに格納
src_df = pd.read_csv(csv_file_path)
print("[likert_p1] Raw DataFrame:")
print(src_df)
