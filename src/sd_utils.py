import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from ordinalcorr import polychoric
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _to_ordinal_int(values):
    """値を順序カテゴリの整数ラベルに変換する。"""
    uniq = np.sort(np.unique(values))
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[v] for v in values], dtype=int)


def _validate_no_constant_columns(vals, col_names):
    """分散0の列がある場合は例外を送出する。"""
    stds = np.nanstd(vals, axis=0)
    zero_cols = [col_names[i] for i, std in enumerate(stds) if std == 0]
    if zero_cols:
        raise ValueError(f"Constant columns are not allowed: {zero_cols}")


def _validate_corr_matrix(corr_mat, context):
    """相関行列の数値妥当性を確認する。"""
    if not np.all(np.isfinite(corr_mat)):
        raise ValueError(f"{context}: correlation matrix contains NaN or Inf.")


def _category_probabilities(vals):
    """各列のカテゴリ確率を返す。"""
    probs_list = []
    for i in range(vals.shape[1]):
        ord_col = _to_ordinal_int(vals[:, i])
        counts = np.bincount(ord_col)
        probs = counts / counts.sum()
        probs_list.append(probs)
    return probs_list


def _simulate_ordinal_data(n_samples, probs_list, rng):
    """カテゴリ確率に従って順序カテゴリデータを生成する。"""
    n_vars = len(probs_list)
    sim_ord = np.empty((n_samples, n_vars), dtype=int)
    for j, probs in enumerate(probs_list):
        sim_ord[:, j] = rng.choice(len(probs), size=n_samples, p=probs)
    return sim_ord


def _ordinal_to_latent_scores(vals):
    """順序カテゴリ値を潜在連続変数の期待値に写像する。"""
    n_samples, n_vars = vals.shape
    latent = np.empty((n_samples, n_vars), dtype=float)
    eps = 1e-7

    for j in range(n_vars):
        ord_col = _to_ordinal_int(vals[:, j])
        counts = np.bincount(ord_col)
        probs = counts / counts.sum()

        cum_probs = np.cumsum(probs)
        thresholds = norm.ppf(np.clip(cum_probs[:-1], eps, 1 - eps))
        lower = np.concatenate(([-np.inf], thresholds))
        upper = np.concatenate((thresholds, [np.inf]))

        den = np.clip(norm.cdf(upper) - norm.cdf(lower), eps, None)
        category_means = (norm.pdf(lower) - norm.pdf(upper)) / den
        z = category_means[ord_col]

        z_std = np.std(z)
        if z_std == 0:
            raise ValueError("Failed to compute latent scores due to zero variance category mapping.")
        latent[:, j] = (z - np.mean(z)) / z_std

    return latent


def _compute_corr_matrix(vals, corr="pearson"):
    """指定した相関法で相関行列を返す。
    Args:
        vals (np.ndarray): データ行列（欠損値は含まないこと）
        corr (str): "pearson" または "polychoric"
    Returns:
        np.ndarray: 相関行列
    """
    if corr == "pearson":
        standard_vals = StandardScaler().fit_transform(vals)
        corr_mat = np.corrcoef(standard_vals, rowvar=False)
        _validate_corr_matrix(corr_mat, "pearson")
        return corr_mat

    if corr == "polychoric":
        n_vars = vals.shape[1]
        corr_mat = np.eye(n_vars, dtype=float)

        # polychoricは1対の順序変数を受け取るため、ペアワイズに計算する
        ord_cols = [_to_ordinal_int(vals[:, i]) for i in range(n_vars)]
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                rho = polychoric(ord_cols[i], ord_cols[j])
                corr_mat[i, j] = rho
                corr_mat[j, i] = rho
        _validate_corr_matrix(corr_mat, "polychoric")
        return corr_mat

    raise ValueError("corr must be 'pearson' or 'polychoric'.")


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


def run_parallel_analysis(src_df, tar_cols, corr="pearson", n_iter=500, percentile=95, random_state=0):
    """
    併行分析を実行し、推奨因子数と比較表を返す。
    Args:
        src_df (pd.DataFrame): 元のデータフレーム
        tar_cols (list): 因子分析に使用するカラム名のリスト
        corr (str): 相関の種類。"pearson" / "polychoric"
        n_iter (int): ランダムデータのシミュレーション回数
        percentile (float): クリティカル値を決定するためのパーセンタイル
        random_state (int): 乱数シード
    """
    # Arg check
    if corr not in ["pearson", "polychoric"]:
        raise ValueError("corr must be 'pearson' or 'polychoric'.")

    try:
        vals_df = src_df[tar_cols].dropna()
        if vals_df.empty:
            raise ValueError("No valid rows were found after dropping missing values.")

        vals = vals_df.values
        n_samples, n_vars = vals.shape
        if n_samples < 3:
            raise ValueError("At least 3 samples are required for parallel analysis.")
        if n_vars < 2:
            raise ValueError("At least 2 variables are required for parallel analysis.")

        _validate_no_constant_columns(vals, tar_cols)

        obs_corr = _compute_corr_matrix(vals, corr=corr)
        obs_eigs = np.sort(np.linalg.eigvalsh(obs_corr))[::-1]

        rng = np.random.default_rng(random_state)
        random_eigs = np.empty((n_iter, n_vars), dtype=float)

        if corr == "polychoric":
            probs_list = _category_probabilities(vals)

        # tqdmを使用して進捗バーを表示する
        for i in tqdm(range(n_iter), desc="Simulating random data"):
            if corr == "polychoric":
                simulated = _simulate_ordinal_data(n_samples, probs_list, rng)
                sim_corr = _compute_corr_matrix(simulated, corr="polychoric")
            else:
                simulated = rng.standard_normal((n_samples, n_vars))
                sim_corr = np.corrcoef(simulated, rowvar=False)

            random_eigs[i, :] = np.sort(np.linalg.eigvalsh(sim_corr))[::-1]

        crit_eigs = np.percentile(random_eigs, percentile, axis=0)
        retain_mask = obs_eigs > crit_eigs
        n_factors = int(retain_mask.sum())

        comp_df = pd.DataFrame(
            {
                "Obs": obs_eigs,
                f"Rnd{percentile}": crit_eigs,
                "Dif": obs_eigs - crit_eigs,
                "Ret": np.where(retain_mask, "Y", "N"),
            }
        )
        comp_df.index = np.arange(1, n_vars + 1)
        comp_df.index.name = "F"
        return n_factors, comp_df
    except Exception as e:
        raise ValueError(f"Parallel analysis failed ({corr}): {e}") from e


def print_parallel_analysis_summary(
    src_df, tar_cols, corr="pearson", n_iter=500, percentile=95, random_state=0, digits=2
):
    """平行分析の結果を表示用文字列として返す。"""
    n_factors, comp_df = run_parallel_analysis(
        src_df,
        tar_cols,
        corr=corr,
        n_iter=n_iter,
        percentile=percentile,
        random_state=random_state,
    )
    summary_lines = [
        f"Recommended factors: {n_factors}",
        "",
        comp_df.round(digits).to_string(),
    ]
    return n_factors, "\n".join(summary_lines)


def factor_analysis(src_df, tar_cols, factor_names, rotation="No rotation", corr="pearson"):
    """因子分析を実行し、指定回転を適用して因子負荷量と因子得点を返す関数
    corr="polychoric" を選択することでPolychoric相関行列を用いた因子分析を実行できる。
    Args:
        src_df (pd.DataFrame): 元のデータフレーム
        tar_cols (list): 因子分析に使用するカラム名のリスト
        factor_names (list): 因子名のリスト（例: ["因子1", "因子2", "因子3"]）
        rotation (str): 回転法。"No rotation" / "varimax" / "promax"
        corr (str): 相関の種類。"pearson" / "polychoric"
    Returns:
        tuple: (rotated_loading_df, factor_score_df, promax_msg)
            rotated_loading_df (pd.DataFrame): 因子負荷量
            factor_score_df (pd.DataFrame): 因子得点
            corr_df (pd.DataFrame or None): promax回転の因子相関行列（rotation="promax"の場合のみ）
    """
    try:
        n_factors = len(factor_names)
        vals_df = src_df[tar_cols].dropna()
        vals = vals_df.values
        standard_vals = StandardScaler().fit_transform(vals)
        if corr not in ["pearson", "polychoric"]:
            raise ValueError("corr must be 'pearson' or 'polychoric'.")

        _validate_no_constant_columns(vals, tar_cols)

        if rotation == "No rotation":
            fa_rotation = None
        else:
            fa_rotation = rotation

        is_corr_matrix = corr == "polychoric"
        fa = FactorAnalyzer(n_factors=n_factors, rotation=fa_rotation, method="minres", is_corr_matrix=is_corr_matrix)
        if is_corr_matrix:
            corr_matrix = _compute_corr_matrix(vals, corr="polychoric")
            fa.fit(corr_matrix)
        else:
            corr_matrix = np.corrcoef(standard_vals, rowvar=False)
            _validate_corr_matrix(corr_matrix, "pearson")
            fa.fit(standard_vals)

        rotated_loadings = fa.loadings_
        if is_corr_matrix:
            # 順序カテゴリを潜在連続変数に写像してから回帰法で因子得点を計算
            latent_vals = _ordinal_to_latent_scores(vals)
            inv_corr = np.linalg.pinv(corr_matrix)
            score_coef = inv_corr @ rotated_loadings
            score_coef = score_coef @ np.linalg.pinv(rotated_loadings.T @ inv_corr @ rotated_loadings)
            factor_scores = latent_vals @ score_coef
        else:
            factor_scores = fa.transform(standard_vals)

        rotated_loading_df = pd.DataFrame(rotated_loadings, index=tar_cols, columns=factor_names)
        valid_index = vals_df.index
        factor_score_df = pd.DataFrame(factor_scores, columns=factor_names, index=valid_index)
        # promax回転の場合、相関係数を取得
        if rotation == "promax":
            # これは因子負荷行列のplotの横に表示するときに使用される
            corr_df = pd.DataFrame(fa.phi_, index=factor_names, columns=factor_names).round(2)

        return rotated_loading_df, factor_score_df, corr_df if rotation == "promax" else None
    except Exception as e:
        raise ValueError(f"Factor analysis failed ({corr}): {e}") from e
