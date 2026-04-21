import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

corr_df = src_df[tar_cols].corr()

print("形容詞対どうしの相関行列:")
print(corr_df.round(3))

# --- 固有値・寄与率をDataFrameで管理 ---
# 相関行列は対称行列なので eigvalsh を使う
eigenvalues = np.sort(np.linalg.eigvalsh(corr_df.values))[::-1]

eig_df = pd.DataFrame({"固有値": eigenvalues})
eig_df.index = np.arange(1, len(eig_df) + 1)
eig_df.index.name = "因子候補"
eig_df["寄与率"] = eig_df["固有値"] / eig_df["固有値"].sum()
eig_df["累積寄与率"] = eig_df["寄与率"].cumsum()

print("\n固有値・寄与率・累積寄与率:")
print(eig_df.round(4))

plt.imshow(corr_df.values, aspect="equal", vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="相関係数")
scale_labels = [
    f"{m.group(1)}-{m.group(2)}" if (m := re.search(r"評価.\(1\)(.+)-(.+)\(7\)", c)) else c for c in tar_cols
]
plt.xticks(range(len(scale_labels)), scale_labels, rotation=90, fontsize=8)
plt.yticks(range(len(scale_labels)), scale_labels, rotation=0, fontsize=8)

for y in range(corr_df.shape[0]):
    for x in range(corr_df.shape[1]):
        plt.text(x, y, f"{corr_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=6)

plt.title("形容詞対どうしの相関行列")
plt.gcf().canvas.manager.set_window_title("Correlation Matrix")
plt.tight_layout()
plt.show()
