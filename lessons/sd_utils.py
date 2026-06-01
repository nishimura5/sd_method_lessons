"""SD法レッスン用ユーティリティAPI。

このモジュールは、レッスンスクリプトから利用する共通APIを提供します。

公開API:
    get_csv_path(tar_file_name, tar_dir="../sample_data")
        読み込み用CSVファイルの絶対パスを返す。
    set_csv_path(tar_file_name, tar_dir="~/Desktop")
        保存用CSVファイルのパスを返す。
    set_japanese_font()
        pandasとmatplotlibの日本語表示設定を行う。
    factor_analysis_with_varimax(src_df, tar_cols, factor_names)
        Varimax回転つき因子分析を実行し、因子負荷量と因子得点を返す。
    compute_eigenvalues(src_df, tar_cols)
        Pearson相関行列の固有値を降順で返す。
    compute_cronbach_alpha(src_df, tar_cols, reverse_cols=None, scale_min=1, scale_max=7)
        Cronbachのα係数を返す。

データ前提:
    因子分析と固有値算出で指定するtar_colsは、数値列である必要があります。
"""

import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


def get_csv_path(tar_file_name, tar_dir="../sample_data"):
    """読み込み用CSVファイルの絶対パスを返す。

    指定されたファイル名を、sd_utils.pyから見たディレクトリに結合して
    絶対パスに解決します。デフォルトでは`../sample_data`を参照します。

    Args:
        tar_file_name (str): 読み込むCSVファイル名。
        tar_dir (str): CSVファイルが置かれているディレクトリ。
            sd_utils.pyからの相対パス、または結合可能なパスを指定します。

    Returns:
        str: CSVファイルの絶対パス。

    Raises:
        FileNotFoundError: 解決後のCSVファイルが存在しない場合。

    Example:
        >>> csv_file_path = get_csv_path("sample_sd.csv")
        >>> src_df = pd.read_csv(csv_file_path)
    """
    # このファイルが置かれているディレクトリをcurrent_dirに格納
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = os.path.abspath(os.path.join(current_dir, tar_dir))
    # データが入っているファイルパスをtar_pathに格納
    tar_path = os.path.join(resolved_dir, tar_file_name)
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"CSV file not found at path: {tar_path}")
    return tar_path


def set_csv_path(tar_file_name, tar_dir="~/Desktop"):
    """保存用CSVファイルのパスを返す。

    指定された保存先ディレクトリとファイル名を結合して、保存用パスを返します。
    この関数はパスを作成するだけで、CSVファイル自体は作成しません。

    Args:
        tar_file_name (str): 保存先CSVファイル名。
        tar_dir (str): 保存先ディレクトリ。`~`はユーザーホームに展開されます。

    Returns:
        str: 保存先CSVファイルのパス。

    Raises:
        FileNotFoundError: 保存先ディレクトリが存在しない場合。

    Example:
        >>> output_path = set_csv_path("factor_loadings.csv")
        >>> rotated_loading_df.to_csv(output_path, encoding="utf-8-sig")
    """
    resolved_dir = os.path.expanduser(tar_dir)
    if not os.path.exists(resolved_dir):
        raise FileNotFoundError(f"Directory not found: {resolved_dir}")
    # For saving CSV file to a specified directory (default: Desktop)
    tar_path = os.path.join(resolved_dir, tar_file_name)
    return tar_path


def set_japanese_font():
    """pandasとmatplotlibの日本語表示設定を行う。

    Args:
        なし。

    Returns:
        None

    Side Effects:
        pandasの`display.unicode.east_asian_width`をTrueに設定します。
        macOSではmatplotlibのフォントを`Hiragino Sans`に設定します。
        macOS以外ではmatplotlibのフォントを`Yu Gothic`に設定します。

    Example:
        >>> set_japanese_font()
    """
    # Enable better alignment for Japanese characters in DataFrame display
    pd.set_option("display.unicode.east_asian_width", True)

    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Darwin":  # macOS
        plt.rcParams["font.family"] = "Hiragino Sans"
    else:
        plt.rcParams["font.family"] = "Yu Gothic"


def factor_analysis_with_varimax(src_df, tar_cols, factor_names):
    """Varimax回転つき因子分析を実行する。

    `tar_cols`で指定した列を標準化し、scikit-learnのFactorAnalysisで
    因子分析を実行します。因子数は`factor_names`の要素数で決まります。

    Args:
        src_df (pd.DataFrame): 入力データ。
        tar_cols (list[str]): 因子分析に使用する数値列名のリスト。
        factor_names (list[str]): 出力する因子名のリスト。
            例: ["因子1", "因子2", "因子3"]

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            (rotated_loading_df, factor_score_df)を返します。

            rotated_loading_df:
                indexは`tar_cols`、columnsは`factor_names`です。
                値はVarimax回転後の因子負荷量です。
            factor_score_df:
                indexは`src_df.index`、columnsは`factor_names`です。
                値は各行の因子得点です。

    Raises:
        KeyError: `tar_cols`に存在しない列名が含まれる場合。
        ValueError: 非数値データ、欠損値、データ不足、因子数不整合などで
            scikit-learnの処理に失敗した場合。

    Example:
        >>> scale_cols = [col for col in src_df.columns if "-" in col]
        >>> factor_names = ["Factor1", "Factor2"]
        >>> loading_df, score_df = factor_analysis_with_varimax(
        ...     src_df,
        ...     scale_cols,
        ...     factor_names,
        ... )
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


def compute_eigenvalues(src_df, tar_cols):
    """Pearson相関行列の固有値を降順で返す。

    `tar_cols`で指定した列から欠損値を含む行を除外し、標準化後に
    Pearson相関行列を作成します。その固有値を大きい順に並べて返します。
    因子数を検討するための参考値として利用できます。

    Args:
        src_df (pd.DataFrame): 入力データ。
        tar_cols (list[str]): 固有値算出に使用する数値列名のリスト。

    Returns:
        np.ndarray: Pearson相関行列の固有値。降順にソート済み。

    Raises:
        KeyError: `tar_cols`に存在しない列名が含まれる場合。
        ValueError: 非数値データ、欠損除外後のデータ不足などで
            numpyまたはscikit-learnの処理に失敗した場合。

    Example:
        >>> scale_cols = [col for col in src_df.columns if "-" in col]
        >>> eigenvalues = compute_eigenvalues(src_df, scale_cols)
        >>> print(eigenvalues)
    """
    vals = src_df[tar_cols].dropna().values
    standard_vals = StandardScaler().fit_transform(vals)
    corr_matrix = np.corrcoef(standard_vals, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
    return eigenvalues


def compute_cronbach_alpha(src_df, tar_cols, reverse_cols=None, scale_min=1, scale_max=7):
    """Cronbachのα係数を返す。

    `tar_cols`で指定した項目列から欠損値を含む行を除外し、各項目の分散と
    合計得点の分散からCronbachのα係数を計算します。`reverse_cols`で指定した
    反転項目は、`scale_min + scale_max - 値`に変換してから計算します。

    Args:
        src_df (pd.DataFrame): 入力データ。
        tar_cols (list[str]): α係数算出に使用する項目列名のリスト。
        reverse_cols (list[str] | str | None): 反転項目として処理する列名。
            単一の列名文字列、列名リスト、またはNoneを指定します。
        scale_min (float): 尺度の最小値。デフォルトは1。
        scale_max (float): 尺度の最大値。デフォルトは7。

    Returns:
        float: Cronbachのα係数。

    Raises:
        KeyError: `tar_cols`に存在しない列名が含まれる場合。
        ValueError: 項目数、データ数、数値変換、分散などの条件を満たさない場合。

    Example:
        >>> scale_cols = [col for col in src_df.columns if "-" in col]
        >>> reverse_cols = ["たかい-やすい"]
        >>> alpha = compute_cronbach_alpha(src_df, scale_cols, reverse_cols=reverse_cols)
        >>> print(round(alpha, 3))
    """
    n_items = len(tar_cols)
    if n_items < 2:
        raise ValueError("At least 2 items are required to compute Cronbach's alpha.")

    if reverse_cols is None:
        reverse_cols = []
    elif isinstance(reverse_cols, str):
        reverse_cols = [reverse_cols]
    else:
        reverse_cols = list(reverse_cols)

    missing_reverse_cols = [col for col in reverse_cols if col not in tar_cols]
    if missing_reverse_cols:
        raise ValueError(f"reverse_cols must be included in tar_cols: {missing_reverse_cols}")

    try:
        scale_min = float(scale_min)
        scale_max = float(scale_max)
    except (TypeError, ValueError) as e:
        raise ValueError("scale_min and scale_max must be numeric.") from e
    if not np.all(np.isfinite([scale_min, scale_max])):
        raise ValueError("scale_min and scale_max must be finite values.")
    if scale_min >= scale_max:
        raise ValueError("scale_min must be smaller than scale_max.")

    item_df = src_df[tar_cols].dropna()
    if len(item_df) < 2:
        raise ValueError("At least 2 valid rows are required after dropping missing values.")

    try:
        numeric_df = item_df.astype(float)
    except ValueError as e:
        raise ValueError("All target columns must be numeric to compute Cronbach's alpha.") from e

    if reverse_cols:
        numeric_df = numeric_df.copy()
        numeric_df.loc[:, reverse_cols] = scale_min + scale_max - numeric_df.loc[:, reverse_cols]

    item_variances = numeric_df.var(axis=0, ddof=1)
    total_scores = numeric_df.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    if not np.all(np.isfinite(item_variances)) or not np.isfinite(total_variance):
        raise ValueError("Cronbach's alpha could not be computed because variance contains NaN or Inf.")
    if total_variance == 0:
        raise ValueError("Total score variance must be greater than 0 to compute Cronbach's alpha.")

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)
