import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

scale_cols = [col for col in src_df.columns if "-" in col]

# 因子の名称を定義
factor_names = ["因子1", "因子2"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, scale_cols, factor_names)

factor_score_df["stimulus_id"] = src_df.loc[src_df.index, "stimulus_id"].values
# Mean factor scores by stimulus (representative positions)
stimulus_factor_df = factor_score_df.groupby("stimulus_id", as_index=True).mean()
print(f"刺激ごとの{len(factor_names)}因子得点平均:")
print(stimulus_factor_df.round(3))

# PCAによる次元削減とプロット
stimulus_factor_std = StandardScaler().fit_transform(stimulus_factor_df.values)
pca = PCA(n_components=2, random_state=0)
stimulus_pca_2d = pca.fit_transform(stimulus_factor_std)

stimulus_pca_df = pd.DataFrame(
    stimulus_pca_2d,
    index=stimulus_factor_df.index,
    columns=["PC1", "PC2"],
)

print("\nPCA 2次元座標（刺激）:")
print(stimulus_pca_df.round(3))

# Annotate original factor axes (Factor1-3) on the PCA plot
factor_axis_vectors_2d = pca.components_.T

# 以下はグラフ描画
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)
plt.scatter(stimulus_pca_df["PC1"], stimulus_pca_df["PC2"], s=80)

for stimulus_code, row in stimulus_pca_df.iterrows():
    plt.text(row["PC1"] + 0.03, row["PC2"] + 0.03, stimulus_code, fontsize=10)

arrow_scale = 1.5
for i, feature_name in enumerate(factor_names):
    x = factor_axis_vectors_2d[i, 0] * arrow_scale
    y = factor_axis_vectors_2d[i, 1] * arrow_scale
    plt.arrow(
        0,
        0,
        x,
        y,
        color="tab:red",
        width=0.005,
        head_width=0.08,
        length_includes_head=True,
        alpha=0.8,
    )
    plt.text(x * 1.08, y * 1.08, feature_name, color="tab:red", fontsize=10)

pc1_ratio = pca.explained_variance_ratio_[0] * 100
pc2_ratio = pca.explained_variance_ratio_[1] * 100
plt.xlabel(f"PC1 ({pc1_ratio:.1f}%)")
plt.ylabel(f"PC2 ({pc2_ratio:.1f}%)")
plt.title("刺激の位置関係（PCA 2次元 + Factor軸アノテーション）")
plt.gcf().canvas.manager.set_window_title("PCA Plot")
plt.tight_layout()
plt.show()
