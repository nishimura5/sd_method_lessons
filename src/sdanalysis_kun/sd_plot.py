from .mpl_setup import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
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
            ax_loadings.text(x, y, f"{loading_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=12)
    ax_loadings.set_title(title if not show_corr else f"{title}: Factor Loadings")
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
                ax_corr.text(x, y, f"{promax_corr_df.iat[y, x]:.2f}", ha="center", va="center", fontsize=12)
        ax_corr.set_title("Promax Factor Correlations")

    plt.gcf().canvas.manager.set_window_title("Factor Loading Matrix")
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 下部10%をキャプション用テキストのために予約
    plt.show()


def _add_sd_ellipse(ax, points, color):
    """Draw a one-standard-deviation covariance ellipse for a set of 2D points."""
    if len(points) < 2:
        return

    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    center = points.mean(axis=0)
    ellipse = Ellipse(
        xy=center,
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.2,
        alpha=0.18,
        zorder=1,
    )
    ax.add_patch(ellipse)


def _draw_stimulus_coordinates(fig, ax, coordinate_df, x_column, y_column, title, stimulus_level=None):
    """Draw 2D stimulus coordinates and return their Matplotlib pick targets."""
    pick_targets = {}
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    if stimulus_level is None:
        points = ax.scatter(coordinate_df[x_column], coordinate_df[y_column], s=20, picker=5)
        pick_targets[points] = list(coordinate_df.index)
        for stimulus_code, row in coordinate_df.iterrows():
            ax.text(row[x_column] + 0.03, row[y_column] + 0.03, stimulus_code, fontsize=10)
    else:
        stimulus_values = coordinate_df.index.get_level_values(stimulus_level)
        grouped_coordinate_df = coordinate_df.assign(_stimulus=stimulus_values).groupby("_stimulus", sort=False)
        groups = list(grouped_coordinate_df)
        color_map = plt.get_cmap("tab20", max(len(groups), 1))

        for group_index, (stimulus_code, group_df) in enumerate(groups):
            color = color_map(group_index)
            points = group_df[[x_column, y_column]].to_numpy()
            center = points.mean(axis=0)
            _add_sd_ellipse(ax, points, color)
            point = ax.scatter(
                center[0],
                center[1],
                s=65,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
                picker=5,
            )
            pick_targets[point] = [stimulus_code]
            ax.text(center[0] + 0.03, center[1] + 0.03, str(stimulus_code), fontsize=10, color=color)

        fig.text(
            0.01,
            0.01,
            "Point: stimulus centroid; ellipse: within-stimulus 1-SD covariance ellipse "
            "(respondent variability; where available)",
            fontsize=8,
        )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title)
    return pick_targets


def _draw_pca(fig, ax, stimulus_factor_df, factor_names, title, stimulus_level=None):
    """Draw factor scores in PCA space on an existing Matplotlib figure."""
    stimulus_factor_std = StandardScaler().fit_transform(stimulus_factor_df[factor_names].values)
    pca = PCA(n_components=2, random_state=0)
    stimulus_pca_2d = pca.fit_transform(stimulus_factor_std)
    factor_axis_vectors_2d = pca.components_.T

    stimulus_pca_df = pd.DataFrame(
        stimulus_pca_2d,
        index=stimulus_factor_df.index,
        columns=["PC1", "PC2"],
    )
    pick_targets = _draw_stimulus_coordinates(
        fig,
        ax,
        stimulus_pca_df,
        "PC1",
        "PC2",
        title,
        stimulus_level,
    )

    arrow_scale = 1.5
    for i, feature_name in enumerate(factor_names):
        x = factor_axis_vectors_2d[i, 0] * arrow_scale
        y = factor_axis_vectors_2d[i, 1] * arrow_scale
        ax.arrow(
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
        ax.text(x * 1.08, y * 1.08, feature_name, color="tab:red", fontsize=10)

    pc1_ratio = pca.explained_variance_ratio_[0] * 100
    pc2_ratio = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_ratio:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.1f}%)")
    fig.tight_layout(rect=[0, 0.04, 1, 1] if stimulus_level is not None else None)
    return pick_targets


def create_pca_map_figure(stimulus_factor_df, factor_names, title, stimulus_level=None):
    """Create a PCA figure that can be embedded in a GUI canvas."""
    fig = Figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    fig.stimulus_pick_targets = _draw_pca(fig, ax, stimulus_factor_df, factor_names, title, stimulus_level)
    return fig


def create_factor_map_figure(stimulus_factor_df, x_factor, y_factor, title, stimulus_level=None):
    """Create a two-factor score figure that can be embedded in a GUI canvas."""
    fig = Figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    fig.stimulus_pick_targets = _draw_stimulus_coordinates(
        fig,
        ax,
        stimulus_factor_df[[x_factor, y_factor]],
        x_factor,
        y_factor,
        title,
        stimulus_level,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1] if stimulus_level is not None else None)
    return fig
