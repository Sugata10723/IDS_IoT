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

    def __init__(self):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_UNSW_NB15.json')
        self.config = self.load_config()
        self.DATA_TRAINING_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/UNSW_NB15_testing-set.csv" # 元データが逆
        self.DATA_TEST_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/UNSW_NB15_training-set.csv" # 元データが逆
        self.data_train = None
        self.data_test = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def preprocess(self, data):
        # グローバル変数を変更しないようにコピー
        data = data.copy()
        # 必要ない列を削除
        data = data.drop(columns=self.config['unwanted_columns'])
        # labelとデータを分離 
        labels = data['label'].values # Numpy Array
        data = data.drop(columns=['label']) # Pandas DataFrame
        
        return data, labels

    def load_data(self):
        self.data_train = pd.read_csv(self.DATA_TRAINING_FILE_PATH)
        self.data_test = pd.read_csv(self.DATA_TEST_FILE_PATH)

        self.X_train, self.y_train = self.preprocess(self.data_train)
        self.X_test, self.y_test = self.preprocess(self.data_test)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config

        



