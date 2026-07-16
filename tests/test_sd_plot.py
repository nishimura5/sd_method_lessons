import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from sdanalysis_kun.sd_plot import plot_factor_loadings, plot_pca


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
        assert all(text.get_fontsize() == 10 for text in ax.texts)

    plt.close("all")


def test_plot_pca_uses_a_new_figure_instead_of_overwriting_current_figure(monkeypatch):
    plt.close("all")
    existing_fig, existing_ax = plt.subplots()
    existing_ax.set_title("Factor Loading Matrix")
    monkeypatch.setattr(plt, "show", lambda: None)
    scores = pd.DataFrame(
        {
            "Factor 1": [-1.0, 0.0, 1.0],
            "Factor 2": [0.5, -1.0, 0.5],
        },
        index=["A", "B", "C"],
    )

    plot_pca(scores, list(scores.columns), title="PCA")

    assert len(plt.get_fignums()) == 2
    assert existing_ax.get_title() == "Factor Loading Matrix"
    assert len(existing_ax.collections) == 0
    assert plt.gcf() is not existing_fig
    assert plt.gca().get_title() == "PCA"

    plt.close("all")
