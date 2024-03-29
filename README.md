# 変更履歴
- 2023/11/17: READMEの概要を作成
- 2023/11/23: 補足を追加
- 2023/12/18: データセットを追記
- 2024/01/15: コードレビューのため情報を追加

# 概要
- 研究の目的:IoTデバイスの通信ログを入力に侵入の検知を行うアルゴリズムを提案した[Fusion-based anomaly detection system using modified isolation forest for internet of things](https://link.springer.com/article/10.1007/s12652-022-04393-9)の追試を行うためのコード
- コードの概要:
    - main.ipynb: 実行プログラム
    - dataset.py: データセットとconfigを読み込むクラス
    - preprocessor.py: データセットのスケーリングや欠損データの補充、不必要な特徴量の削除を行うクラス
    - ids.py: 予測を行うクラス
    - trainer.py: アルゴリズムのトレーニングを行うクラス
    - experiment.py: 実験を行うクラス
    - congig: json形式で書かれているconfigファイルを保存するディレクトリ
    - data: データセットを保存するディレクトリ
    - result: 結果を保存するためのディレクトリ

# 環境設定
- 必要なソフトウェア/ライブラリ: 後ほど追記します

# 実行方法 
- 設定ファイル: configディレクトリにあるjson形式のファイルが設定ファイルになります。各項目の説明は以下のとおりです。
    - name_dataset:読み込むデータセットの名前   
    - num_rows:読み込むデータセットの数
    - unwanted_columns:トレーニングに必要ないカラムを設定
    - categorical_columns:カテゴリカルカラムを設定

# データ
- 使用データ:
    - IoTNetworkIntrusionDataset: 原論文中には存在しないデータセット
    - UNSW_NB15: 論文中でも使用されたデータセット 
    - NSL_KDD: 原論文中で使用されたデータセット
- データの準備: 
    - 以下の参考文献のリンクから取得してください

# 参考文献
- [Fusion-based anomaly detection system using modified isolation forest for internet of things](https://link.springer.com/article/10.1007/s12652-022-04393-9)
- [Network_intrusion_dataset](https://sites.google.com/view/iot-network-intrusion-dataset/home)
- [UNSW_NB15](後ほど追記します)
- [KDDCUP99](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [NSL_KDD](https://www.unb.ca/cic/datasets/nsl.html)
