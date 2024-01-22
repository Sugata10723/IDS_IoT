import pandas as pd
import os
import json
import numpy as np 
########################################
# Dataset_UNSW_NB15
# 本来であれば49個ある特徴量が46個しかない(dstip, srcip, Stime, Ltimeがなく、rateが追加されている)
# labelとattack_catを除くと44個の特徴量
# カテゴリ変数は5個
# 数値変数は39個
# training-setとtesting-setに分かれているが、これは逆である
# training-set: 175341行
# testing-set: 82332行
########################################
class Dataset_UNSW_NB15:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, nrows):
        self.nrows = nrows
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_UNSW_NB15.json')
        self.DATA_TRAINING_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/UNSW_NB15_testing-set.csv" # 元データが逆
        self.DATA_TEST_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/UNSW_NB15_training-set.csv" # 元データが逆

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def preprocess(self, data, config):
        # グローバル変数を変更しないようにコピー
        data = data.copy()
        nrows = self.nrows
        # データをサンプリング
        if nrows > data.shape[0]:
            nrows = data.shape[0]
        data = data.sample(n=nrows)
        # 必要ない列を削除
        data = data.drop(columns=config['unwanted_columns'])
        # インデックスをリセット
        data = data.reset_index()
        # labelとデータを分離 
        labels = data['label'].values # Numpy Array
        data = data.drop(columns=['label']) # Pandas DataFrame
        
        return data, labels

    def load_data(self, config):
        data_train = pd.read_csv(self.DATA_TRAINING_FILE_PATH)
        data_test = pd.read_csv(self.DATA_TEST_FILE_PATH)

        X_train, y_train = self.preprocess(data_train, config)
        X_test, y_test = self.preprocess(data_test, config)

        return X_train, X_test, y_train, y_test

    def get_data(self):
        config = self.load_config()
        X_train, X_test, y_train, y_test = self.load_data(config)
        
        return X_train, X_test, y_train, y_test, config

        



