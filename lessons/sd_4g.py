import matplotlib.pyplot as plt
import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 集計表を作成
melted_df = src_df.melt(id_vars=["対象物コード"], value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="対象物コード", values="value", aggfunc="mean")
heatmap_df = heatmap_df.reindex(index=tar_cols)

# グラフ描画
plt.imshow(heatmap_df.values, aspect="auto", vmin=1, vmax=7, cmap="coolwarm")

# Axes labels
plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90)
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

# Annotate cells
for y in range(heatmap_df.shape[0]):
    for x in range(heatmap_df.shape[1]):
        value = heatmap_df.iat[y, x]
        if pd.notna(value):
            plt.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.colorbar(label="平均評定")

plt.title("評定の平均値 (全対象物)")
plt.gcf().canvas.manager.set_window_title("Rating Overview (All Objects)")
plt.tight_layout()
plt.show()
