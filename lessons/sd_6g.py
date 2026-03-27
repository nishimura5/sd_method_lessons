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
rotated_loading_df = rotated_loading_df.drop(columns=["max_abs_loading", "best_factor"])

# 以下はグラフ描画
# Rename index
rotated_loading_df.index = rotated_loading_df.index.str.replace(
    r"評価\.\(\d\)(.*?)-(.*)\(\d\)", r"\1 -- \2", regex=True
)
plt.imshow(rotated_loading_df.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
plt.xticks(range(rotated_loading_df.shape[1]), rotated_loading_df.columns, rotation=0)
plt.yticks(range(rotated_loading_df.shape[0]), rotated_loading_df.index)

for y in range(rotated_loading_df.shape[0]):
    for x in range(rotated_loading_df.shape[1]):
        plt.text(x, y, f"{rotated_loading_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=8)

plt.title(f"因子負荷行列({len(factor_names)}因子モデル・バリマックス回転)")
plt.gcf().canvas.manager.set_window_title("Factor Loading Matrix")
plt.tight_layout()
plt.show()
