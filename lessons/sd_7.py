import matplotlib.pyplot as plt
import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 因子の名称を定義
factor_names = ["因子1", "因子2", "因子3"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, tar_cols, factor_names)

# Sort factors
rotated_loading_df["max_abs_loading"] = rotated_loading_df.abs().max(axis=1)
rotated_loading_df["best_factor"] = rotated_loading_df.abs().idxmax(axis=1)
rotated_loading_df = rotated_loading_df.sort_values(["best_factor", "max_abs_loading"], ascending=[True, False])
sorted_scale_cols = rotated_loading_df.index.tolist()

# 因子負荷が負の形容詞対を反転させるためのリストを作成
invert_list = [
    bool(rotated_loading_df.loc[col, rotated_loading_df.loc[col, "best_factor"]] < 0) for col in sorted_scale_cols
]

# 集計表を作成
melted_df = src_df.melt(id_vars=["対象物コード"], value_vars=tar_cols, var_name="形容詞対")
heatmap_df = melted_df.pivot_table(index="形容詞対", columns="対象物コード", values="value", aggfunc="mean")
heatmap_df = heatmap_df.reindex(index=sorted_scale_cols)
# ここで表示用にリネーム
heatmap_df.index = heatmap_df.index.str.replace(r"評価\.\(\d\)(.*?)-(.*)\(\d\)", r"\1 -- \2", regex=True)

# invert_listに基づいて、因子負荷が負の形容詞対を反転する
invert_mask = pd.Series(invert_list, index=heatmap_df.index)
heatmap_df.loc[invert_mask] = 8 - heatmap_df.loc[invert_mask]
heatmap_df.index = [
    f"{right} -- {left}" if invert else label
    for label, invert in zip(heatmap_df.index, invert_mask)
    for left, right in [label.split(" -- ", 1)]
]

# 因子負荷が最も大きい因子と反転の有無を表示
best_factor_series = pd.Series(
    rotated_loading_df.loc[sorted_scale_cols, "best_factor"].values,
    index=heatmap_df.index,
)
for heatmap_idx, invert, factor in zip(heatmap_df.index, invert_mask, best_factor_series):
    print(f"  [{factor}] {heatmap_idx} {'(反転)' if invert else ''}")

# 以下はグラフ描画
plt.imshow(heatmap_df.values, aspect="auto", vmin=1, vmax=7, cmap="coolwarm")

plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90)
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

for y in range(heatmap_df.shape[0]):
    for x in range(heatmap_df.shape[1]):
        value = heatmap_df.iat[y, x]
        if pd.notna(value):
            plt.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.colorbar(label="平均評定")

plt.title("評定の平均値（因子でソート）")
plt.gcf().canvas.manager.set_window_title("Factor-Sorted Rating")
plt.tight_layout()
plt.show()
