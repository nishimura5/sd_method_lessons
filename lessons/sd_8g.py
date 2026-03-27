import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sd_utils

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 因子の名称を定義
factor_names = ["因子1", "因子2", "因子3"]

rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, tar_cols, factor_names)

factor_score_df["対象物コード"] = src_df.loc[src_df.index, "対象物コード"].values
# Mean factor scores by object (representative positions)
object_factor_df = factor_score_df.groupby("対象物コード", as_index=True).mean()
print(f"対象物ごとの{len(factor_names)}因子得点平均:")
print(object_factor_df.round(3))

# PCAによる次元削減とプロット
object_factor_std = StandardScaler().fit_transform(object_factor_df.values)
pca = PCA(n_components=2, random_state=0)
object_pca_2d = pca.fit_transform(object_factor_std)

object_pca_df = pd.DataFrame(
    object_pca_2d,
    index=object_factor_df.index,
    columns=["PC1", "PC2"],
)

print("\nPCA 2次元座標（対象物）:")
print(object_pca_df.round(3))

# Annotate original factor axes (Factor1-3) on the PCA plot
loadings = pca.components_.T

# 以下はグラフ描画
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)
plt.scatter(object_pca_df["PC1"], object_pca_df["PC2"], s=80)

for object_code, row in object_pca_df.iterrows():
    plt.text(row["PC1"] + 0.03, row["PC2"] + 0.03, object_code, fontsize=10)

arrow_scale = 1.5
for i, feature_name in enumerate(factor_names):
    x = loadings[i, 0] * arrow_scale
    y = loadings[i, 1] * arrow_scale
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
plt.title("対象物の位置関係（PCA 2次元 + Factor軸アノテーション）")
plt.gcf().canvas.manager.set_window_title("PCA Plot")
plt.tight_layout()
plt.show()
