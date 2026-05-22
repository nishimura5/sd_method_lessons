import re

import matplotlib.pyplot as plt
import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

scale_cols = [col for col in src_df.columns if "-" in col]

# 全刺激の回答件数のsumを集計
melted_df = src_df.melt(id_vars=["respondent_id", "stimulus_id"], value_vars=scale_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="value", aggfunc="size", fill_value=0)

# 標準偏差を計算
std_series = src_df[scale_cols].std()

# heat_map_dfの右に標準偏差の列を追加
print_df = heatmap_df.copy()
print_df["標準偏差"] = std_series

# 結果を表示
print("評定の件数 (全刺激合計)と標準偏差:")
print(print_df)

# Create ordered lists of left and right labels
tar_cols = heatmap_df.index.tolist()
left_order = []
right_order = []
for col in tar_cols:
    m = re.search(r"(.+)-(.+)", col)
    left_order.append(m.group(1))
    right_order.append(m.group(2))

# グラフ描画
plt.imshow(heatmap_df.values, aspect="auto", cmap="GnBu")

# X axis (ratings 1..7)
plt.xlim(-0.5, 6.5)
plt.xticks(range(7), range(1, 8))

# Left Y labels
plt.yticks(range(len(tar_cols)), left_order)

# Annotate cells
for y in range(heatmap_df.shape[0]):
    for x in range(heatmap_df.shape[1]):
        plt.text(x, y, f"{int(heatmap_df.iat[y, x])}", ha="center", va="center", fontsize=9)

# Right Y labels (twin axis)
left_axis = plt.gca()
right_axis = left_axis.twinx()
right_axis.set_ylim(left_axis.get_ylim())
right_axis.set_yticks(range(len(tar_cols)))
right_axis.set_yticklabels(right_order)
right_axis.tick_params(axis="y", length=0)

plt.title("評定の件数 (全刺激合計)")
plt.gcf().canvas.manager.set_window_title("Rating Aggregation")
plt.tight_layout()
plt.show()
