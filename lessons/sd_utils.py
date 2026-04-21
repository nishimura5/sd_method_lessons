import os
import platform

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

def get_csv_path(tar_file_name, tar_dir="../sample_data"):
    # For loading CSV file from a specified directory (default: ../sample_data)
    # このファイルが置かれているディレクトリをcurrent_dirに格納
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = os.path.abspath(os.path.join(current_dir, tar_dir))
    # データが入っているファイルパスをtar_pathに格納
    tar_path = os.path.join(resolved_dir, tar_file_name)
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"CSV file not found at path: {tar_path}")
    return tar_path

def set_csv_path(tar_file_name, tar_dir="~/Desktop"):
    resolved_dir = os.path.expanduser(tar_dir)
    if not os.path.exists(resolved_dir):
        raise FileNotFoundError(f"Directory not found: {resolved_dir}")
    # For saving CSV file to a specified directory (default: Desktop)
    tar_path = os.path.join(resolved_dir, tar_file_name)
    return tar_path


def set_japanese_font():
    # Enable better alignment for Japanese characters in DataFrame display
    pd.set_option("display.unicode.east_asian_width", True)

    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Darwin":  # macOS
        plt.rcParams["font.family"] = "Hiragino Sans"
    else:
        plt.rcParams["font.family"] = "Yu Gothic"


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
