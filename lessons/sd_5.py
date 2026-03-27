import matplotlib.pyplot as plt
import pandas as pd
import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

corr_df = src_df[tar_cols].corr()

print("形容詞対どうしの相関行列:")
print(corr_df.round(3))

plt.imshow(corr_df.values, aspect="equal", vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="相関係数")
scale_labels = [c.replace("評価.(1)", "").replace("(7)", "") for c in tar_cols]
plt.xticks(range(len(scale_labels)), scale_labels, rotation=90, fontsize=8)
plt.yticks(range(len(scale_labels)), scale_labels, rotation=0, fontsize=8)

# Annotate cells
for y in range(corr_df.shape[0]):
    for x in range(corr_df.shape[1]):
        plt.text(x, y, f"{corr_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=6)

plt.title("形容詞対どうしの相関行列")
plt.gcf().canvas.manager.set_window_title("Correlation Matrix")
plt.tight_layout()
plt.show()
