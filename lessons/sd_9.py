import matplotlib.pyplot as plt
import pandas as pd
import sd_utils
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

sd_utils.set_japanese_font()

csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
src_df = pd.read_csv(csv_file_path)

tar_cols = [c for c in src_df.columns if c.startswith("評価.")]

# 因子の名称を定義
factor_names = ["因子1", "因子2", "因子3"]

# Factor scores for each response row
rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, tar_cols, factor_names)
factor_score_df["対象物コード"] = src_df.loc[src_df.index, "対象物コード"].values

# Mean factor scores by object
object_factor_df = factor_score_df.groupby("対象物コード", as_index=True).mean()

# Standardize each factor column before distance calculation and clustering
object_factor_std = StandardScaler().fit_transform(object_factor_df.values)
object_factor_std_df = pd.DataFrame(
    object_factor_std,
    index=object_factor_df.index,
    columns=factor_names,
)
print("\n標準化後の平均因子得点:")
print(object_factor_std_df.round(3))

# Euclidean distance matrix between objects
object_distance_df = pd.DataFrame(
    euclidean_distances(object_factor_std_df.values),
    index=object_factor_std_df.index,
    columns=object_factor_std_df.index,
)
print("\n対象物どうしの距離（Euclidean）:")
print(object_distance_df.round(3))

# Hierarchical clustering with Ward method
linkage_matrix = linkage(object_factor_std_df.values, method="ward")

# Compare candidate numbers of clusters by silhouette score
n_objects = len(object_factor_std_df)
if n_objects < 3:
    raise ValueError("クラスタリングの比較には、少なくとも3個の対象物が必要です。")

max_clusters = min(4, n_objects - 1)
candidate_n_clusters = list(range(2, max_clusters + 1))

score_rows = []
for n_clusters in candidate_n_clusters:
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    score = silhouette_score(object_factor_std_df.values, labels)
    score_rows.append({"クラスタ数": n_clusters, "silhouette": score})

score_df = pd.DataFrame(score_rows).set_index("クラスタ数")
print("\nクラスタ数の候補ごとの silhouette:")
print(score_df.round(3))

best_n_clusters = int(score_df["silhouette"].idxmax())
print(f"\n採用するクラスタ数: {best_n_clusters}")

# Final cluster assignment
final_labels = fcluster(linkage_matrix, t=best_n_clusters, criterion="maxclust")
clustered_object_df = object_factor_df.copy()
clustered_object_df["クラスタ"] = final_labels
clustered_object_df = clustered_object_df.sort_values(["クラスタ"] + factor_names)

print("\n対象物ごとのクラスタ:")
print(clustered_object_df.round(3))

cluster_profile_df = clustered_object_df.groupby("クラスタ")[factor_names].mean()
print("\nクラスタごとの平均因子得点:")
print(cluster_profile_df.round(3))

# 以下はグラフ描画（樹形図 + silhouette 横並び）
fig, (ax_dend, ax_sil) = plt.subplots(1, 2)

# 左：樹形図
dn = dendrogram(
    linkage_matrix,
    labels=object_factor_std_df.index.tolist(),
    ax=ax_dend,
)
for icoord, dcoord in zip(dn["icoord"], dn["dcoord"]):
    x = (icoord[1] + icoord[2]) / 2
    y = dcoord[1]
    ax_dend.annotate(
        f"{y:.2f}",
        xy=(x, y),
        xytext=(5, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
    )
# 採用クラスタ数のカットライン
cut_y = (linkage_matrix[-best_n_clusters, 2] + linkage_matrix[-(best_n_clusters - 1), 2]) / 2
ax_dend.axhline(y=cut_y, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax_dend.text(
    ax_dend.get_xlim()[1],
    cut_y,
    f" n={best_n_clusters}クラスタ",
    color="red",
    va="bottom",
    fontsize=8,
)
ax_dend.set_ylabel("結合距離")
ax_dend.set_title("対象物の樹形図（Ward法）")

# 右：silhouette
ax_sil.bar(score_df.index.astype(str), score_df["silhouette"])
ax_sil.set_xlabel("クラスタ数")
ax_sil.set_ylabel("silhouette")
ax_sil.set_title("クラスタ数候補の比較")

fig.canvas.manager.set_window_title("Dendrogram & Silhouette")
plt.tight_layout()
plt.show()
