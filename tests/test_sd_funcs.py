import numpy as np
import pandas as pd
import pytest

from sdanalysis_kun.sd_funcs import summarize_factor_scores


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
