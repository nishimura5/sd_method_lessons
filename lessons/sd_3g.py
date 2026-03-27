import matplotlib.pyplot as plt
import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

# 集計したい対象物コードを含むデータだけを抽出してtar_dfに格納
target_obj = "o_001"
tar_df = src_df[src_df["対象物コード"] == target_obj].copy()

tar_cols = [c for c in tar_df.columns if c.startswith("評価.")]

# 集計表を作成
melted_df = tar_df.melt(value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="value", aggfunc="size", fill_value=0)
heatmap_df = heatmap_df.reindex(index=tar_cols, columns=range(1, 8), fill_value=0)

print(f"評定の件数 ({target_obj}):")
print(heatmap_df)

# Create ordered lists of left and right labels
tar_cols = heatmap_df.index.tolist()
left_order = []
right_order = []
for col in tar_cols:
    left_label = col.split("(1)")[1].split("-")[0]
    right_label = col.split("-")[1].split("(7)")[0]
    left_order.append(left_label)
    right_order.append(right_label)

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

plt.title(f"評定の件数 ({target_obj})")
plt.gcf().canvas.manager.set_window_title("Rating Aggregation")
plt.tight_layout()
plt.show()
