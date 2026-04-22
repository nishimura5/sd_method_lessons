import re

import matplotlib.pyplot as plt
import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 全objの回答件数のsumを集計
melted_df = src_df.melt(value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="value", aggfunc="size", fill_value=0)
heatmap_df = heatmap_df.reindex(index=tar_cols, columns=range(1, 8), fill_value=0)

print("評定の件数 (全obj合計):")
print(heatmap_df)

# Create ordered lists of left and right labels
tar_cols = heatmap_df.index.tolist()
left_order = []
right_order = []
for col in tar_cols:
    m = re.search(r"評価.\(1\)(.+)-(.+)\(7\)", col)
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

plt.title("評定の件数 (全obj合計)")
plt.gcf().canvas.manager.set_window_title("Rating Aggregation")
plt.tight_layout()
plt.show()
