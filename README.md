# Unbiased Recommender Learning With Relevance-FactorizationMachines

本リポジトリは大学院での研究を元にアルゴリズムの性能実験を行うものです。  
研究の内容はショートverとして<a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/short_paper.md">short_paper.md</a>に記載しております。  
先にこちらをご覧くださいませ。

# 主な使用技術
詳細は、<a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/pyproject.toml">pyproject.toml</a>を参照下さい。
|名称|バージョン|説明|
|:---:|:--------:|:-:|
|Python|3.9|-|
|Docker|20.10.21|コンテナ環境|
|Docker Compose|2.13.0|コンテナ管理|
|Poetry|1.6.1|Pythonパッケージ管理|
|Numpy|1.24.3|行列演算ライブラリ|
|SciPy|1.9.3|科学計算ライブラリ|
|Pandas|2.0.1|表データ管理|
|scikit-learn|1.2.2|機械学習汎用ライブラリ|
|matplotlib|3.7.1|描画ライブラリ|
|seaborn|0.12.2|描画ライブラリ|
|hydra-core|1.3.2|設定管理・アプリケーション構造化|
|pytest|7.4.0|テストライブラリ|

アルゴリズムへの深い理解と実装スキルの向上を目指しているため、コードは主にNumpyとSciPyを使用したオリジナルの実装となっています。

# 使用するデータセット
本研究での実験には、<a href="https://kuairec.com/">KuaiRecデータセット</a>を使用します。使用する背景の詳細は、<a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/short_paper.md">`short_paper.md`</a>の`半合成データを用いた性能実験`の項目を参照してください。

## 扱うcsvファイルの詳細
実験には以下の5つのcsvファイルを使用します。(*実験スクリプトを実行する際は、これらのファイルを/data/kuairec/ディレクトリに格納する必要があります。)

- `small_matrix.csv`: 実験的に収集された、ユーザー数1411人、動画数3327本のフィードバックデータ。評価値行列の密度は約99.6%
- `big_matrix.csv`: 自然に観測された、ユーザー数7176人、動画数10728本のフィードバックデータ
- `item_categories.csv`: 各動画のカテゴリ情報
- `item_daily_features.csv`: 日毎の動画特徴量
- `user_features.csv`: 匿名化されたユーザー特徴量

# ディレクトリ構成

- `conf`: 実験設定関連
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.py">`config.py`</a>: `config.yaml`の型アノテーションクラス
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.yaml">`config.yaml`</a>: 構造化された実験設定
- `data`: 実験データ管理
  - `best_params`: 各モデル・推定量ごとのチューニングパラメータ (json形式)
  -  `Kuairec`: 実験で使用する<a href="https://kuairec.com/">KuaiRecデータセット</a>格納ディレクトリ。(*csvファイル必須)
- `logs`: 実験結果・ログ
  - `.hydra`: hydraの設定ディレクトリ
  - `reslut`: 実験結果管理
    - `img`: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/plot.py">`Visualizer`</a>による画像結果
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/logs/result/metric.csv">`metric.csv`</a>: 詳細な結果指標
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/logs/main.log">`main.log`</a>: 実験ログ
- `src`:  推薦アルゴリズム実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/base.py">`base.py`</a>: 基底クラスの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/fm.py">`fm.py`</a>: Factorization Machines (FM) の実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/mf.py">`mf.py`</a>: Matrix Factorization (MF) の実装
- `test`: ユニットテスト (pytest使用)
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_dataloader.py">`test_dataloader.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/loader.py">`DataLoader`</a>のテスト
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_fm.py">`test_fm.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/fm.py">`FactorizationMachines`</a>のテスト
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_mf.py">`test_mf.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/mf.py">`LogisticMatrixFactorization`</a>のテスト
- `utils`: 汎用的モジュール
  - `dataloader`: データロード関連
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/base.py">`base.py`</a>: 基底クラス
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/loader.py">`loader.py`</a>: 主要なDataLoaderクラス
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_click.py">`_click.py`</a>: 半人工ログデータ生成
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_feature.py">`_feature.py`</a>: 特徴量の生成
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_preperer.py">`_preperer.py`</a>: 学習、評価データの準備
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_kuairec.py">`_kuairec.py`</a>: KuaiRecデータセットのロード
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/evaluate.py">`evaluate.py`</a>: 学習済みの機械学習モデルの評価
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/metrics.py">`metrics.py`</a>: 評価指標の計算
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/model.py">`model.py`</a>: 型アノテーション
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/optimizer.py">`optimizer.py`</a>:  最適化アルゴリズム
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/plot.py">`plot.py`</a>: 実験結果の可視化
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/search_params.py">`search_params.py`</a>: パラメータのチューニング
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/Dockerfile">`Dockerfile`</a>:Docker Imageの設定
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/README.md">`README.md`</a>: レポジトリの詳細な説明
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/docker-compose.yml">`docker-compose.yml`</a>: コンテナ管理の設定
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/main.py">`main.py`</a>: 実験スクリプトの実行
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/poetry.lock">`poetry.lock`</a>: Pythonパッケージの詳細
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/pyproject.toml">`pyproject.toml`</a>: Pythonパッケージ
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/short_paper.md">`short_paper.md`</a>: 実験の概要・背景・実験設定・結果

# 実験設定の詳細

このドキュメントでは、推薦システムの実験設定についての詳細な解説を提供します。設定は[こちらのconfig.yaml](https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.yaml)で確認できます。

## 実験の再現性

- **`seed`**: 実験の再現性を保証するための乱数のシード値

## 半人工データセットの生成設定 (`data_logging_settings`)

- **`data_path`**: KuaiRecデータセットの保存先
- **`train_val_test_ratio`**: データセットの訓練、検証、テストへの分割比率
- **`density`**: 評価値行列の密度
- **`behavior_policy`**: ログデータ生成のアルゴリズム。露出バイアスのみを仮定し、ランダムポリシーを使用

## KuaiRecデータセットのテーブル設定 (`tables`)

### 1. Interactionテーブル

- **`data_path`**: ログデータの保存先
- **`used_features`**: 実験に使用する特徴量のリスト
  - **key**: カラム名
  - **value**: モデルへの入力用データタイプ

### 2. Userテーブル

- **`data_path`**: ユーザー特徴量の保存先
- **`used_features`**: 実験に使用する特徴量のリスト
  - **key**: カラム名
  - **value**: モデルへの入力用データタイプ

### 3. Videoテーブル

#### 3.1 Dailyテーブル

- **`data_path`**: 日別の動画特徴量の保存先
- **`used_features`**: 実験に使用する特徴量のリスト
  - **key**: カラム名
  - **value**: モデルへの入力用データタイプ

#### 3.2 Categoryテーブル

- **`data_path`**: 動画カテゴリー情報の保存先
- **`used_features`**: 実験に使用する特徴量のリスト
  - **key**: カラム名
  - **value**: モデルへの入力用データタイプ

## モデルのハイパーパラメータ設定

- **`is_search_params`**: ハイパーパラメータのチューニングを実験前に行うかどうかのフラグ
- **`model_param_range`**: 各モデルのハイパーパラメータの範囲
  - **`MF` (Matrix Factorization)**
    - **`n_epochs`**: 学習のイテレーション数
    - **`n_factors`**: 潜在因子の数
    - **`lr`**: 学習率
    - **`batch_size`**: バッチサイズ
    - **`clipping`**: 傾向スコアのクリッピング値
  - **`FM` (Factorization Machines)**
    - (同様の設定が含まれています)

---
このドキュメントは、実験設定の詳細を提供するものです。具体的な実装やさらなる詳細については、ソースコードや関連ドキュメントを参照してください。
