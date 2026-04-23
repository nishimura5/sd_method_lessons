import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_factor_loadings(loading_df, title, inverted_rows=None, promax_corr_df=None, caption=""):
    show_corr = promax_corr_df is not None
    fig, axes = plt.subplots(1, 2 if show_corr else 1, figsize=(12, 6) if show_corr else None)

    ax_loadings = axes[0] if show_corr else axes
    ax_loadings.imshow(loading_df.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax_loadings.set_xticks(range(loading_df.shape[1]))
    ax_loadings.set_xticklabels(loading_df.columns, rotation=0)
    y_labels = list(loading_df.index)
    if inverted_rows is not None and len(inverted_rows) == len(y_labels):
        y_labels = [f"{label}*" if inverted else str(label) for label, inverted in zip(y_labels, inverted_rows)]
    ax_loadings.set_yticks(range(loading_df.shape[0]))
    ax_loadings.set_yticklabels(y_labels)
    for y in range(loading_df.shape[0]):
        for x in range(loading_df.shape[1]):
            ax_loadings.text(x, y, f"{loading_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=8)
    ax_loadings.set_title(title if not show_corr else f"{title} - Factor Loadings")
    # caption用のスペースをグラフの下に確保
    # キャプションを追加
    if caption:
        plt.figtext(0.01, 0.01, caption, wrap=True, horizontalalignment="left", fontsize=10)

    if show_corr:
        ax_corr = axes[1]
        ax_corr.imshow(promax_corr_df.values, aspect="equal", vmin=-1, vmax=1, cmap="coolwarm")
        ax_corr.set_xticks(range(promax_corr_df.shape[1]))
        ax_corr.set_xticklabels(promax_corr_df.columns, rotation=0)
        ax_corr.set_yticks(range(promax_corr_df.shape[0]))
        ax_corr.set_yticklabels(promax_corr_df.index)
        for y in range(promax_corr_df.shape[0]):
            for x in range(promax_corr_df.shape[1]):
                ax_corr.text(x, y, f"{promax_corr_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=8)
        ax_corr.set_title("Promax Factor Correlations")

    plt.gcf().canvas.manager.set_window_title("Factor Loading Matrix")
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 下部5%をキャプション用テキストのために予約
    plt.show()



def plot_pca(object_factor_df, factor_names, title):
    object_factor_std = StandardScaler().fit_transform(object_factor_df.values)
    pca = PCA(n_components=2, random_state=0)
    object_pca_2d = pca.fit_transform(object_factor_std)
    factor_axis_vectors_2d = pca.components_.T

    object_pca_df = pd.DataFrame(
        object_pca_2d,
        index=object_factor_df.index,
        columns=["PC1", "PC2"],
    )
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.scatter(object_pca_df["PC1"], object_pca_df["PC2"], s=20)

    for object_code, row in object_pca_df.iterrows():
        plt.text(row["PC1"] + 0.03, row["PC2"] + 0.03, object_code, fontsize=10)

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
    plt.title(title)
    plt.gcf().canvas.manager.set_window_title("PCA Plot")
    plt.tight_layout()
    plt.show()
