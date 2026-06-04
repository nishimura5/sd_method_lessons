# SD Method Lessons

SD法（Semantic Differential法）の基本的な集計から、因子分析、因子得点、刺激のマッピング、クラスタリングまでを、段階的な Python スクリプトで学ぶためのリポジトリです。

このリポジトリの中心は `lessons/` 配下のレッスンコードです。あわせて、同じ分析を GUI で試すための `SDAnalysis-kun` アプリも含まれています。

## このリポジトリでできること

- SD法の CSV データを読み込む
- 刺激・回答者・形容詞対を確認する
- 形容詞対ごとの平均評定、評定分布、標準偏差を集計する
- 形容詞対どうしの相関行列と固有値を確認する
- Varimax 回転つき因子分析を行う
- 因子負荷量に基づいて形容詞対を並べ替え、必要に応じて尺度方向を反転する
- 回答行ごとの因子得点、刺激ごとの平均因子得点を求める
- PCA による刺激マップを描く
- 刺激間距離、階層的クラスタリング、silhouette によるクラスタ数比較を行う
- GUI アプリから CSV を選択し、因子分析・可視化・CSV エクスポートを行う

## ディレクトリ構成

```text
.
├── README.md
├── pyproject.toml
├── sample_data/
│   └── sample_sd.csv
├── lessons/
│   ├── sd_1.py
│   ├── sd_2.py
│   ├── sd_3.py
│   ├── sd_3g.py
│   ├── sd_4.py
│   ├── sd_4g.py
│   ├── sd_5.py
│   ├── sd_6.py
│   ├── sd_6g.py
│   ├── sd_7.py
│   ├── sd_8.py
│   ├── sd_8g.py
│   ├── sd_9.py
│   └── sd_utils.py
├── src/
│   └── sdanalysis_kun/
│       ├── app_sd.py
│       ├── cli.py
│       ├── mpl_setup.py
│       ├── sd_funcs.py
│       ├── sd_plot.py
│       └── tooltip.py
├── build.bat
├── build_mac.zsh
├── app.version
└── LICENSE
```

## 想定データ

レッスンコードは、デフォルトで `sample_data/sample_sd.csv` を読み込みます。

`sample_sd.csv` は次のような構造を想定しています。

```csv
id,respondent_id,datetime,stimulus_id,はやい-おそい,たかい-やすい,...
1,r_001,2026-05-21T09-29-37,s_003,3,2,...
```

主な列は以下です。

| 列 | 内容 |
| --- | --- |
| `id` | 回答行 ID |
| `respondent_id` | 回答者 ID |
| `datetime` | 回答日時 |
| `stimulus_id` | 評価対象の刺激 ID |
| `はやい-おそい` など | SD法の形容詞対。値は数値評定 |

`lessons/` のスクリプトでは、列名に `-` を含む数値列を形容詞対として扱います。

## セットアップ

Python 3.12 以上を想定しています。

このリポジトリでは `pyproject.toml` に依存ライブラリが定義されています。`uv` を使う場合は、リポジトリ直下で次を実行します。

```bash
uv venv
uv pip install -e .
```

主な依存ライブラリは以下です。

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- factor-analyzer
- ordinalcorr

`lessons/` の基礎レッスンは主に pandas、numpy、matplotlib、scikit-learn、scipy を使います。GUI アプリ側では factor-analyzer や ordinalcorr も使います。

## レッスンの実行方法

リポジトリ直下から、次のように各スクリプトを実行します。

```bash
uv run python lessons/sd_1.py
```

例:

```bash
uv run python lessons/sd_3.py
uv run python lessons/sd_3g.py
uv run python lessons/sd_6.py
uv run python lessons/sd_9.py
```

`*_g.py` は、通常版の集計に加えてグラフ表示を行うスクリプトです。

## レッスン一覧

### 1. データ読み込み

| ファイル | 内容 |
| --- | --- |
| `lessons/sd_1.py` | `sample_sd.csv` を読み込み、ファイルパス、行数、DataFrame の内容を表示します。 |
| `lessons/sd_2.py` | `stimulus_id` と `respondent_id` のユニークな値を確認します。 |

### 2. 形容詞対の集計

| ファイル | 内容 |
| --- | --- |
| `lessons/sd_3.py` | 列名に `-` を含む列を形容詞対として抽出し、縦持ちデータへ変換したうえで、刺激ごとの平均評定表を作成します。 |
| `lessons/sd_3g.py` | `sd_3.py` と同じ平均評定表をヒートマップとして表示します。 |
| `lessons/sd_4.py` | 全刺激をまとめて、形容詞対ごとの評定値の件数と標準偏差を表示します。 |
| `lessons/sd_4g.py` | 評定値の件数分布を、左右の形容詞ラベルつきヒートマップとして表示します。 |

### 3. 相関と因子数の検討

| ファイル | 内容 |
| --- | --- |
| `lessons/sd_5.py` | 形容詞対どうしの Pearson 相関行列を求め、固有値、寄与率、累積寄与率を表示します。相関行列のヒートマップも表示します。 |
| `lessons/sd_6.py` | `sd_utils.compute_eigenvalues()` で固有値を確認し、2因子モデルの Varimax 回転つき因子分析を行い、因子負荷行列を表示します。 |
| `lessons/sd_6g.py` | `sd_6.py` の因子負荷行列をヒートマップとして表示します。 |

### 4. 因子負荷量の整理と因子得点

| ファイル | 内容 |
| --- | --- |
| `lessons/sd_7.py` | 因子負荷量の絶対値が最大の因子に基づいて形容詞対を並べ替えます。負の因子負荷を持つ形容詞対は、解釈しやすいように尺度方向を反転して表示します。 |
| `lessons/sd_8.py` | 各回答行の因子得点を計算し、回答者・刺激ごとの因子得点と、刺激ごとの平均因子得点を表示します。 |
| `lessons/sd_8g.py` | 刺激ごとの平均因子得点を標準化し、PCA で 2 次元に配置した刺激マップを表示します。 |

### 5. 刺激の比較・クラスタリング

| ファイル | 内容 |
| --- | --- |
| `lessons/sd_9.py` | 刺激ごとの平均因子得点を標準化し、ユークリッド距離、Ward 法による階層的クラスタリング、silhouette によるクラスタ数比較、クラスタごとの平均因子得点を表示します。樹形図と silhouette 比較グラフも表示します。 |

## `lessons/sd_utils.py` の役割

`sd_utils.py` は、レッスンコードから使う共通処理をまとめた補助モジュールです。

| 関数 | 内容 |
| --- | --- |
| `get_csv_path()` | `sample_data/` 内の CSV ファイルパスを解決します。存在しない場合は `FileNotFoundError` を送出します。 |
| `set_csv_path()` | デスクトップなど、保存先 CSV パスを作成します。 |
| `set_japanese_font()` | pandas の日本語表示幅と matplotlib の日本語フォントを設定します。 |
| `compute_eigenvalues()` | 指定列の Pearson 相関行列から固有値を降順で返します。 |
| `factor_analysis_with_varimax()` | 指定列を標準化し、scikit-learn の `FactorAnalysis` で Varimax 回転つき因子分析を実行します。 |
| `compute_cronbach_alpha()` | 指定項目の Cronbach の α 係数を計算します。反転項目にも対応しています。 |

## レッスンコードを変更するときのポイント

### 別の CSV を使う

`lessons/sd_utils.py` の `get_csv_path()` は、デフォルトで `../sample_data` を参照します。

```python
csv_file_path = sd_utils.get_csv_path("sample_sd.csv")
```

別のファイルを使う場合は、`sample_data/` に CSV を置いてファイル名を変えるか、`get_csv_path()` の引数を変更してください。

```python
csv_file_path = sd_utils.get_csv_path("my_sd_data.csv")
```

### 形容詞対の列名

レッスンコードでは、おおむね次の条件で形容詞対列を抽出しています。

```python
scale_cols = [col for col in src_df.columns if "-" in col]
```

そのため、形容詞対列は `はやい-おそい` のように `-` を含む名前にしておくと、そのまま利用できます。

### 因子数を変更する

`sd_6.py`、`sd_7.py`、`sd_8.py`、`sd_8g.py`、`sd_9.py` では、次のように因子名を定義しています。

```python
factor_names = ["因子1", "因子2"]
```

3因子モデルを試す場合は、次のように変更します。

```python
factor_names = ["因子1", "因子2", "因子3"]
```

### 評定尺度について

`sample_sd.csv` は 1〜7 の 7件法を想定しています。特に `sd_7.py` では、反転表示に `8 - 値` を使っているため、7件法前提の処理になっています。

5件法など別の尺度を使う場合は、反転処理の式を尺度に合わせて変更してください。

## GUI アプリ: SDAnalysis-kun

`src/sdanalysis_kun/` には、Tkinter ベースの GUI アプリが含まれています。レッスンコードで行っている分析を、CSV 選択とボタン操作で試すための補助ツールです。

起動コマンド:

```bash
uv run sdanalysis-kun
```

または:

```bash
uv run sd-method-lessons
```

GUI では主に以下の操作ができます。

- CSV ファイルの選択
- 刺激列の選択
- 回答者列の選択（任意）
- 分析対象の刺激フィルタ
- 形容詞対列の選択
- 形容詞対名を整形する正規表現の指定
- 5件法・7件法の選択
- Pearson 相関または polychoric 相関による併行分析
- 因子数の指定、または併行分析結果（PA）に基づく因子数指定
- Promax 回転、Varimax 回転、無回転の選択
- 因子負荷行列の表示・プロット・CSV 出力
- 刺激ごとの因子得点の表示・CSV 出力
- PCA による刺激マップの表示

## アプリのビルド

配布用アプリを作るための PyInstaller スクリプトも含まれています。

### Windows

```bat
build.bat
```

成功すると、`dist/SDAnalysis-kun.exe` が作成されます。

### macOS

```bash
chmod +x ./build_mac.zsh
./build_mac.zsh
```

成功すると、`dist/SDAnalysis-kun.app` が作成されます。

macOS 版では、環境変数を使って署名や notarization も指定できます。詳細は `build_mac.zsh` のコメントを参照してください。

## 注意事項

- 現在のレッスンコードは `sample_sd.csv` を前提にしています。
- `likert_*.py` 系のファイルは、このリポジトリの現行構成には含まれていません。
- `lessons/` の因子分析は scikit-learn の `FactorAnalysis` を使った Varimax 回転の教材用実装です。
- GUI アプリ側の因子分析は `factor-analyzer` を使い、Promax、Varimax、無回転、Pearson 相関、polychoric 相関に対応しています。
- グラフ表示には matplotlib を使います。実行環境によっては GUI バックエンドや日本語フォントの設定が必要になることがあります。

## ライセンス

Apache License 2.0
