import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler


def get_japanese_monospace_font():
    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Darwin":  # macOS
        return "Osaka-Mono"
    else:  # Windows
        return "MS Gothic"


def get_japanese_proportional_font():
    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Darwin":  # macOS
        return "Hiragino Sans"
    else:  # Windows
        return "Yu Gothic"


def set_japanese_font():
    pd.set_option("display.unicode.east_asian_width", True)
    # plt font settings
    plt.rcParams["font.family"] = get_japanese_proportional_font()


def run_parallel_analysis(src_df, tar_cols, n_iter=500, percentile=95, random_state=0):
    """併行分析を実行し、推奨因子数と比較表を返す。"""
    vals_df = src_df[tar_cols].dropna()
    if vals_df.empty:
        raise ValueError("No valid rows were found after dropping missing values.")

    vals = vals_df.values
    n_samples, n_vars = vals.shape
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for parallel analysis.")
    if n_vars < 2:
        raise ValueError("At least 2 variables are required for parallel analysis.")

    standard_vals = StandardScaler().fit_transform(vals)
    obs_corr = np.corrcoef(standard_vals, rowvar=False)
    obs_eigs = np.sort(np.linalg.eigvalsh(obs_corr))[::-1]

    rng = np.random.default_rng(random_state)
    random_eigs = np.empty((n_iter, n_vars), dtype=float)
    for i in range(n_iter):
        simulated = rng.standard_normal((n_samples, n_vars))
        sim_corr = np.corrcoef(simulated, rowvar=False)
        random_eigs[i, :] = np.sort(np.linalg.eigvalsh(sim_corr))[::-1]

    crit_eigs = np.percentile(random_eigs, percentile, axis=0)
    retain_mask = obs_eigs > crit_eigs
    n_factors = int(retain_mask.sum())

    comp_df = pd.DataFrame(
        {
            "Observed": obs_eigs,
            f"Rand.({percentile}%)": crit_eigs,
            "Diff": obs_eigs - crit_eigs,
            "Retain": np.where(retain_mask, "Yes", "No"),
        }
    )
    comp_df.index = np.arange(1, n_vars + 1)
    comp_df.index.name = "Factor"
    return n_factors, comp_df


def print_parallel_analysis_summary(src_df, tar_cols, n_iter=500, percentile=95, random_state=0, digits=3):
    """併行分析の結果を表示用文字列として返す。"""
    n_factors, comp_df = run_parallel_analysis(
        src_df,
        tar_cols,
        n_iter=n_iter,
        percentile=percentile,
        random_state=random_state,
    )
    summary_lines = [
        f"Recommended factors (Parallel Analysis): {n_factors}",
        "",
        comp_df.round(digits).to_string(),
    ]
    return n_factors, "\n".join(summary_lines)


def factor_analysis(src_df, tar_cols, factor_names, rotation="No rotation"):
    """因子分析を実行し、指定回転を適用して因子負荷量と因子得点を返す関数
    Args:
        src_df (pd.DataFrame): 元のデータフレーム
        tar_cols (list): 因子分析に使用するカラム名のリスト
        factor_names (list): 因子名のリスト（例: ["因子1", "因子2", "因子3"]）
        rotation (str): 回転法。"No rotation" / "varimax" / "promax"
    Returns:
        tuple: (rotated_loading_df, factor_score_df, promax_msg)
            rotated_loading_df (pd.DataFrame): 因子負荷量
            factor_score_df (pd.DataFrame): 因子得点
            corr_df (pd.DataFrame or None): promax回転の因子相関行列（rotation="promax"の場合のみ）
    """
    n_factors = len(factor_names)
    vals = src_df[tar_cols].dropna().values
    standard_vals = StandardScaler().fit_transform(vals)
    if rotation == "No rotation":
        fa_rotation = None
    else:
        fa_rotation = rotation

    fa = FactorAnalyzer(n_factors=n_factors, rotation=fa_rotation, method="minres")
    fa.fit(standard_vals)
    rotated_loadings = fa.loadings_
    factor_scores = fa.transform(standard_vals)
    rotated_loading_df = pd.DataFrame(rotated_loadings, index=tar_cols, columns=factor_names)
    valid_index = src_df[tar_cols].dropna().index
    factor_score_df = pd.DataFrame(factor_scores, columns=factor_names, index=valid_index)
    # promax回転の場合、相関係数を取得
    if rotation == "promax":
        # これは因子負荷行列のplotの横に表示するときに使用される
        corr_df = pd.DataFrame(fa.phi_, index=factor_names, columns=factor_names).round(2)

    return rotated_loading_df, factor_score_df, corr_df if rotation == "promax" else None
