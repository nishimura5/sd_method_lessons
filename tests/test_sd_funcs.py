import numpy as np
import pandas as pd
import pytest
from factor_analyzer import FactorAnalyzer

from sdanalysis_kun.sd_funcs import factor_analysis, summarize_factor_scores


def test_summarize_factor_scores_calculates_mean_and_sample_sd():
    factor_score_df = pd.DataFrame(
        {
            "Factor1": [1.0, 3.0, 5.0],
            "Factor2": [2.0, 4.0, 8.0],
            "stimulus": ["A", "A", "B"],
        }
    )

    mean_df, summary_df = summarize_factor_scores(factor_score_df, ["stimulus"], ["Factor1", "Factor2"])

    assert mean_df.loc["A", "Factor1"] == pytest.approx(2.0)
    assert mean_df.loc["A", "Factor2"] == pytest.approx(3.0)
    assert summary_df.columns.tolist() == ["Factor1 Mean", "Factor1 SD", "Factor2 Mean", "Factor2 SD"]
    assert summary_df.loc["A", "Factor1 Mean"] == pytest.approx(2.0)
    assert summary_df.loc["A", "Factor1 SD"] == pytest.approx(np.sqrt(2.0))
    assert summary_df.loc["A", "Factor2 Mean"] == pytest.approx(3.0)
    assert summary_df.loc["A", "Factor2 SD"] == pytest.approx(np.sqrt(2.0))
    assert np.isnan(summary_df.loc["B", "Factor1 SD"])


def test_factor_analysis_returns_promax_pattern_structure_and_aligned_correlations():
    rng = np.random.default_rng(42)
    latent = rng.multivariate_normal(
        [0.0, 0.0, 0.0],
        [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
        size=500,
    )
    population_loadings = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.0],
            [0.7, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.1, 0.6, 0.1],
            [0.0, 0.5, 0.2],
            [0.1, 0.0, 0.55],
            [0.0, 0.1, 0.5],
            [0.1, 0.1, 0.45],
        ]
    )
    values = latent @ population_loadings.T + rng.normal(scale=0.5, size=(500, 9))
    columns = [f"item_{i}" for i in range(values.shape[1])]
    factor_names = ["Factor 1", "Factor 2", "Factor 3"]
    source = pd.DataFrame(values, columns=columns)

    pattern_loading_df, structure_loading_df, factor_score_df, factor_corr_df = factor_analysis(
        source, columns, factor_names, rotation="promax"
    )

    fitted = FactorAnalyzer(n_factors=3, rotation="promax", method="minres").fit(values)
    expected_phi = np.linalg.lstsq(fitted.loadings_, fitted.structure_, rcond=None)[0]
    expected_phi = (expected_phi + expected_phi.T) / 2
    np.fill_diagonal(expected_phi, 1.0)

    # This data causes factor-analyzer 0.5.1 to change the factor order.
    # Its phi_ remains in the pre-sort order, while loadings_ and structure_
    # use the post-sort order.
    assert not np.allclose(fitted.phi_, expected_phi)
    np.testing.assert_allclose(pattern_loading_df.to_numpy(), fitted.loadings_)
    np.testing.assert_allclose(structure_loading_df.to_numpy(), fitted.structure_)
    np.testing.assert_allclose(factor_corr_df.to_numpy(), expected_phi)
    np.testing.assert_allclose(
        structure_loading_df.to_numpy(),
        pattern_loading_df.to_numpy() @ factor_corr_df.to_numpy(),
    )
    assert factor_score_df.shape == (len(source), len(factor_names))


def test_factor_analysis_returns_no_structure_or_factor_correlations_for_varimax():
    rng = np.random.default_rng(7)
    columns = [f"item_{i}" for i in range(5)]
    source = pd.DataFrame(rng.normal(size=(100, len(columns))), columns=columns)

    pattern_loading_df, structure_loading_df, factor_score_df, factor_corr_df = factor_analysis(
        source, columns, ["Factor 1", "Factor 2"], rotation="varimax"
    )

    assert pattern_loading_df.shape == (len(columns), 2)
    assert structure_loading_df is None
    assert factor_score_df.shape == (len(source), 2)
    assert factor_corr_df is None
