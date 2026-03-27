import platform

import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


def get_japanese_monospace_font():
    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Darwin":  # macOS
        return "Osaka-Mono"
    else:  # Windows
        return "MS Gothic"


def set_japanese_font():
    pd.set_option("display.unicode.east_asian_width", True)


def factor_analysis_with_varimax(src_df, tar_cols, factor_names):
    """因子分析を実行し、Varimax回転を適用して因子負荷量と因子得点を返す関数
    Args:
        src_df (pd.DataFrame): 元のデータフレーム
        tar_cols (list): 因子分析に使用するカラム名のリスト
        factor_names (list): 因子名のリスト（例: ["因子1", "因子2", "因子3"]）
    Returns:
        tuple: (rotated_loading_df, factor_score_df)
            rotated_loading_df (pd.DataFrame): 因子負荷量
            factor_score_df (pd.DataFrame): 因子得点
    """
    n_factors = len(factor_names)
    vals = src_df[tar_cols].values
    standard_vals = StandardScaler().fit_transform(vals)
    fa_varimax = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=0)
    fa_varimax.fit(standard_vals)
    rotated_loadings = fa_varimax.components_.T
    rotated_loading_df = pd.DataFrame(rotated_loadings, index=tar_cols, columns=factor_names)
    factor_scores = fa_varimax.transform(standard_vals)
    factor_score_df = pd.DataFrame(factor_scores, columns=factor_names, index=src_df.index)
    return rotated_loading_df, factor_score_df
