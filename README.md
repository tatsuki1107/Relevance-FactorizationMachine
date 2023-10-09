# Relevance-FactorizationMachines

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

# ディレクトリ構成

- `conf`: 実験設定を管理するモジュール
  
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.py">`config.py`</a>: 読み取り専用の<a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.yaml">`config.yaml`</a>の型アノテーションクラスの実装
    
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/config.yaml">`config.yaml`</a>: 構造化された実験設定
  
- `data`: 実験に使用するデータを管理するモジュール
  
  - `best_params`: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/search_params.py">`search_params.py`</a>でチューニングしたパラメータを管理するディレクトリ。モデル(FM, MF)と推定量(Ideal, IPS, Naive)ごとにチューニングしたパラメータをjsonで格納している。
    
  -  `Kuairec`: 実験で使用する<a href="https://kuairec.com/">KuaiRecデータセット</a>を格納するディレクトリ。(*実験を開始するにはこちらのディレクトリにcsvファイルを格納する必要があります。)
- `logs`: 実験結果とログを管理するモジュール
  
  - `.hydra`: スクリプトを実行した際のhydraの詳細な設定ディレクトリ
    
  - `reslut`: 実験結果を管理するディレクトリ
    
    - `img`: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/plot.py">`Visualizer`</a>クラスを実行した際の実験結果の画像を管理するディレクトリ
      
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/logs/result/metric.csv">`metric.csv`</a>: 詳細な結果の指標を格納したcsvファイル
- `src`: 推薦アルゴリズムをまとめたディレクトリ
  
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/base.py">`base.py`</a>: 基底クラスの実装
    
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/fm.py">`fm.py`</a>: Factorization Machines (FM) の実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/mf.py">`mf.py`</a>: Matrix Factorization (MF) の実装
- `test`: pytestを用いたユニットテストを行うモジュール
  
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_dataloader.py">`test_dataloader.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/loader.py">`DataLoader`</a>クラスのテストの実装
    
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_fm.py">`test_fm.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/fm.py">`FactorizationMachines`</a>クラスのテストの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/test/test_mf.py">`test_mf.py`</a>: <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/src/mf.py">`LogisticMatrixFactorization`</a>クラスのテストの実装
- `utils`: 汎用的に扱うモジュール郡
  - `dataloader`: データロードのためのパイプライン
  
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/base.py">`base.py`</a>: データロードの基底クラスの実装
    
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/loader.py">`loader.py`</a>: main.pyから呼び出すDataLoaderクラスの実装
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_click.py">`_click.py`</a>: 半人工ログデータからクリックデータを生成するSemiSyntheticLogDataGeneratorクラスの実装
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_feature.py">`_feature.py`</a>: 特徴量を生成するFeatureGeneratorクラスの実装
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_preperer.py">`_preperer.py`</a>: MFとFMを用いて学習、評価を行うためにデータを準備するDatasetPreparerクラスの実装
    - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/dataloader/_kuairec.py">`_kuairec.py`</a>: kuairecデータセットのcsvファイルをロードするKuaiRecCSVLoaderクラスの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/evaluate.py">`evaluate.py`</a>: 学習済みの機械学習モデルを用いて評価を行うEvaluatorクラスの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/metrics.py">`metrics.py`</a>: 評価指標を計算する関数の実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/model.py">`model.py`</a>: データのロード中に使用する型アノテーションクラスの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/optimizer.py">`optimizer.py`</a>: 非凸目的関数に使用する最適化アルゴリズムの確率的勾配降下法とAdamの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/plot.py">`plot.py`</a>: 実験結果をグラフとして可視化するVisualizerクラスの実装
  - <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/utils/search_params.py">`search_params.py`</a>: 最適化するパラメータのチューニングを実行する関数の実装
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/Dockerfile">`Dockerfile`</a>:使用するDocker Imageの記述
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/README.md">`README.md`</a>: 本レポジトリの詳細な説明を記述したマークダウンファイル
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/docker-compose.yml">`docker-compose.yml`</a>: コンテナ管理の設定を構造化したymlファイル
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/main.py">`main.py`</a>: 実験スクリプトを実行するPythonファイル
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/poetry.lock">`poetry.lock`</a>: Pythonパッケージの詳細なバージョンが記述されたlockファイル
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/pyproject.toml">`pyproject.toml`</a>: Pythonパッケージのバージョンが記述されたtomlファイル
- <a href="https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/short_paper.md">`short_paper.md`</a>: 実験の概要、背景、実験設定、そして結果を論文形式で記述したマークダウンファイル。

