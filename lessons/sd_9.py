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

scale_cols = [col for col in src_df.columns if "-" in col]

# 因子の名称を定義
factor_names = ["因子1", "因子2"]

# Factor scores for each response row
rotated_loading_df, factor_score_df = sd_utils.factor_analysis_with_varimax(src_df, scale_cols, factor_names)
factor_score_df["stimulus_id"] = src_df.loc[src_df.index, "stimulus_id"].values

# Mean factor scores by stimulus
stimulus_factor_df = factor_score_df.groupby("stimulus_id", as_index=True).mean()

# Standardize each factor column before distance calculation and clustering
stimulus_factor_std = StandardScaler().fit_transform(stimulus_factor_df.values)
stimulus_factor_std_df = pd.DataFrame(
    stimulus_factor_std,
    index=stimulus_factor_df.index,
    columns=factor_names,
)
print("\n標準化後の平均因子得点:")
print(stimulus_factor_std_df.round(3))

# Euclidean distance matrix between stimuli
stimulus_distance_df = pd.DataFrame(
    euclidean_distances(stimulus_factor_std_df.values),
    index=stimulus_factor_std_df.index,
    columns=stimulus_factor_std_df.index,
)
print("\n刺激どうしの距離（Euclidean）:")
print(stimulus_distance_df.round(3))

# Hierarchical clustering with Ward method
linkage_matrix = linkage(stimulus_factor_std_df.values, method="ward")

# Compare candidate numbers of clusters by silhouette score
n_stimuli = len(stimulus_factor_std_df)
if n_stimuli < 3:
    raise ValueError("クラスタリングの比較には、少なくとも3個の刺激が必要です。")

max_clusters = min(4, n_stimuli - 1)
candidate_n_clusters = list(range(2, max_clusters + 1))

score_rows = []
for n_clusters in candidate_n_clusters:
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    score = silhouette_score(stimulus_factor_std_df.values, labels)
    score_rows.append({"クラスタ数": n_clusters, "silhouette": score})

score_df = pd.DataFrame(score_rows).set_index("クラスタ数")
print("\nクラスタ数の候補ごとの silhouette:")
print(score_df.round(3))

best_n_clusters = int(score_df["silhouette"].idxmax())
print(f"\n採用するクラスタ数: {best_n_clusters}")

# Final cluster assignment
final_labels = fcluster(linkage_matrix, t=best_n_clusters, criterion="maxclust")
clustered_stimulus_df = stimulus_factor_df.copy()
clustered_stimulus_df["クラスタ"] = final_labels
clustered_stimulus_df = clustered_stimulus_df.sort_values(["クラスタ"] + factor_names)

print("\n刺激ごとのクラスタ:")
print(clustered_stimulus_df.round(3))

cluster_profile_df = clustered_stimulus_df.groupby("クラスタ")[factor_names].mean()
print("\nクラスタごとの平均因子得点:")
print(cluster_profile_df.round(3))

# 以下はグラフ描画（樹形図 + silhouette 横並び）
fig, (ax_dend, ax_sil) = plt.subplots(1, 2)

# 左：樹形図
dn = dendrogram(
    linkage_matrix,
    labels=stimulus_factor_std_df.index.tolist(),
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
ax_dend.set_title("刺激の樹形図（Ward法）")

# 右：silhouette
ax_sil.bar(score_df.index.astype(str), score_df["silhouette"])
ax_sil.set_xlabel("クラスタ数")
ax_sil.set_ylabel("silhouette")
ax_sil.set_title("クラスタ数候補の比較")

fig.canvas.manager.set_window_title("Dendrogram & Silhouette")
plt.tight_layout()
plt.show()
