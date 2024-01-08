import pandas as pd
import os
import json
import numpy as np 
########################################
# Dataset_UNSW_NB15
# 本来であれば49個ある特徴量が44個しかない
# training-setとtesting-setに分かれているが、これは逆である
# training-set: 175341行
# testing-set: 82332行
########################################
class Dataset_UNSW_NB15:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, nrows):
        self.nrows = nrows
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

    def split_ip(self, ip):
        return list(map(int, ip.split('.')))

    def preprocess(self, data):
        data = data.copy()
        # 指定した行数だけ読み込む
        #data = data.iloc[:self.config['nrows']]

        # 不要なカラムを削除
        data = data.drop(columns=self.config['unwanted_columns'])

        data = data.reset_index() 
        labels = data['label'] # Pandas Series
        data = data.drop(columns=['label']) # Pandas DataFrame
        return data, labels

    def load_data(self):
        self.data_train = pd.read_csv(self.DATA_TRAINING_FILE_PATH)
        self.data_test = pd.read_csv(self.DATA_TEST_FILE_PATH)

        # 指定した行数だけ読み込む
        if self.nrows > self.X_train.shape[0]:
            self.nrows = self.X_train.shape[0]
        self.X_train = self.X_train.iloc[:self.nrows]
        if int(self.nrows * 0.3) > self.X_test.shape[0]:
            self.nrows = self.X_test.shape[0]
        self.X_test = self.X_test.iloc[:int(self.nrows * 0.3)]

        self.X_train, self.y_train = self.preprocess(self.data_train)
        self.X_test, self.y_test = self.preprocess(self.data_test)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config

        



