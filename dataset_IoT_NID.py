import pandas as pd
from zipfile import ZipFile
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#################################################################################
# Dataset_IoT_Network_Intrusion_Dataset
# 元はAttackが多いデータセット
# 最終的な変換後の特徴量数は87
# Src_IPとDst_IPを整数値に変換
# infがある行があるので削除する
#################################################################################

class Dataset_IoT_NID:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, nrows):
        self.nrows = nrows
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_IoT_NID.json')
        self.config = self.load_config()
        self.DATA_CSV_FILE_PATH = f"{self.BASE_DIR}/data/IoT_Network_Intrusion_Dataset/IoT_Network_Intrusion_Dataset.csv"
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = None

        self.load_data()
        self.split_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
        self.y_train = self.y_train.values # Pandas Series -> NumPy Array
        self.y_test = self.y_test.values # Pandas Series -> NumPy Array

    def preprocess(self, data):
        # グローバル変数を変更しないようにコピー
        data = data.copy()
        nrows = self.nrows
        # 指定した行数だけ読み込む
        if nrows > data.shape[0]:
            nrows = data.shape[0]
        data = data.sample(n=nrows)

        # labelを正常:0, 攻撃:1に変換
        data['Label'] = data['Label'].map({'Anomaly': 1, 'Normal': 0}) # Pandas Series
        #　不用なカラムを削除
        data.drop(columns=self.config["unwanted_columns"], inplace=True)
        # infのある行を削除
        inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
        if inf_rows.sum() > 0:
            print(f"Number of rows with inf: {inf_rows.sum()}")
        data = data[~inf_rows]
        # インデックスのリセット
        data.reset_index(drop=True, inplace=True)
        # ラベルを分割
        labels = data['Label'] # Pandas Series 
        data = data.drop('Label', axis=1) # Pandas DataFrame

        return data, labels

    def load_data(self):
        self.data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        self.data, self.labels = self.preprocess(self.data)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config
        

        



