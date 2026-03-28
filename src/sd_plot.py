import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_factor_loadings(loading_df, factor_names, title):
    plt.imshow(loading_df.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(loading_df.shape[1]), loading_df.columns, rotation=0)
    plt.yticks(range(loading_df.shape[0]), loading_df.index)
    for y in range(loading_df.shape[0]):
        for x in range(loading_df.shape[1]):
            plt.text(x, y, f"{loading_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=8)

    plt.title(title)
    plt.gcf().canvas.manager.set_window_title("Factor Loading Matrix")
    plt.tight_layout()
    plt.show()


def plot_pca(object_factor_df, factor_names, title):
    object_factor_std = StandardScaler().fit_transform(object_factor_df.values)
    pca = PCA(n_components=2, random_state=0)
    object_pca_2d = pca.fit_transform(object_factor_std)
    loadings = pca.components_.T

    object_pca_df = pd.DataFrame(
        object_pca_2d,
        index=object_factor_df.index,
        columns=["PC1", "PC2"],
    )
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
    plt.title(title)
    plt.gcf().canvas.manager.set_window_title("PCA Plot")
    plt.tight_layout()
    plt.show()
