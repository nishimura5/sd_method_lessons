import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sdanalysis_kun.sd_plot import create_factor_map_figure, create_pca_map_figure, plot_factor_loadings


def test_plot_factor_loadings_uses_larger_heatmap_value_labels(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)
    loadings = pd.DataFrame(
        {"Factor 1": [0.75, -0.25], "Factor 2": [0.10, 0.80]},
        index=["Item 1", "Item 2"],
    )
    correlations = pd.DataFrame(
        {"Factor 1": [1.0, 0.30], "Factor 2": [0.30, 1.0]},
        index=["Factor 1", "Factor 2"],
    )

    plot_factor_loadings(loadings, title="Loadings", promax_corr_df=correlations)

    for ax in plt.gcf().axes:
        assert len(ax.texts) == 4
        assert all(text.get_fontsize() == 12 for text in ax.texts)

    plt.close("all")


def test_create_pca_map_figure_summarizes_respondent_points_by_stimulus():
    plt.close("all")
    index = pd.MultiIndex.from_tuples(
        [("R1", "A"), ("R2", "A"), ("R3", "A"), ("R1", "B"), ("R2", "B"), ("R3", "B")],
        names=["respondent", "stimulus"],
    )
    scores = pd.DataFrame(
        {
            "Factor 1": [-1.0, -0.5, -0.8, 0.7, 1.2, 0.9],
            "Factor 2": [0.0, 0.6, -0.4, -0.5, 0.3, 0.8],
        },
        index=index,
    )

    fig = create_pca_map_figure(scores, list(scores.columns), title="PCA", stimulus_level="stimulus")

    ax = fig.axes[0]
    stimulus_labels = {text.get_text() for text in ax.texts} - set(scores.columns)
    assert stimulus_labels == {"A", "B"}
    assert sum(len(collection.get_offsets()) for collection in ax.collections) == 2
    assert sum(isinstance(patch, Ellipse) for patch in ax.patches) == 2
    caption = fig.texts[0].get_text()
    assert "stimulus centroid" in caption
    assert "within-stimulus 1-SD covariance ellipse" in caption
    assert sorted(stimulus_ids[0] for stimulus_ids in fig.stimulus_pick_targets.values()) == ["A", "B"]

    plt.close("all")


def test_create_pca_map_figure_does_not_register_a_pyplot_window():
    plt.close("all")
    scores = pd.DataFrame(
        {
            "Factor 1": [-1.0, 0.0, 1.0],
            "Factor 2": [0.5, -1.0, 0.5],
        },
        index=["A", "B", "C"],
    )

    fig = create_pca_map_figure(scores, list(scores.columns), title="Embedded PCA")

    assert plt.get_fignums() == []
    assert fig.axes[0].get_title() == "Embedded PCA"
    assert len(fig.axes[0].collections) == 1
    assert list(fig.stimulus_pick_targets.values()) == [["A", "B", "C"]]


def test_create_factor_map_figure_uses_selected_factor_scores_without_pca():
    plt.close("all")
    scores = pd.DataFrame(
        {
            "Factor1": [-1.2, 0.4, 1.8],
            "Factor2": [9.0, 8.0, 7.0],
            "Factor3": [0.3, -0.7, 1.1],
        },
        index=["A", "B", "C"],
    )

    fig = create_factor_map_figure(scores, "Factor1", "Factor3", title="Selected Factors")

    ax = fig.axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), scores[["Factor1", "Factor3"]].to_numpy())
    assert ax.get_xlabel() == "Factor1"
    assert ax.get_ylabel() == "Factor3"
    assert ax.get_title() == "Selected Factors"
    assert list(fig.stimulus_pick_targets.values()) == [["A", "B", "C"]]
    assert plt.get_fignums() == []


def test_create_factor_map_figure_summarizes_respondents_with_shared_map_drawing():
    index = pd.MultiIndex.from_tuples(
        [("R1", "A"), ("R2", "A"), ("R3", "A"), ("R1", "B"), ("R2", "B"), ("R3", "B")],
        names=["respondent", "stimulus"],
    )
    scores = pd.DataFrame(
        {
            "Factor1": [-1.0, -0.5, -0.8, 0.7, 1.2, 0.9],
            "Factor2": [0.0, 0.6, -0.4, -0.5, 0.3, 0.8],
        },
        index=index,
    )

    fig = create_factor_map_figure(
        scores,
        "Factor1",
        "Factor2",
        title="Factor Axes",
        stimulus_level="stimulus",
    )

    ax = fig.axes[0]
    assert {text.get_text() for text in ax.texts} == {"A", "B"}
    assert sum(len(collection.get_offsets()) for collection in ax.collections) == 2
    assert sum(isinstance(patch, Ellipse) for patch in ax.patches) == 2
    assert sorted(stimulus_ids[0] for stimulus_ids in fig.stimulus_pick_targets.values()) == ["A", "B"]
