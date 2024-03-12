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

    def __init__(self):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_IoT_NID.json')
        self.DATA_CSV_FILE_PATH = f"{self.BASE_DIR}/data/IoTID20/IoT_Network_Intrusion_Dataset.csv"

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def split_data(self, data, labels, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        y_train = y_train.values # Pandas Series -> NumPy Array
        y_test = y_test.values # Pandas Series -> NumPy Array

        return X_train, X_test, y_train, y_test

    def preprocess(self, data, config):
        # グローバル変数を変更しないようにコピー
        data = data.copy()
        # labelを正常:0, 攻撃:1に変換
        data['Label'] = data['Label'].map({'Anomaly': 1, 'Normal': 0}) # Pandas Series
        #　不用なカラムを削除
        data.drop(columns=config["unwanted_columns"], inplace=True)
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

    def load_data(self, config):
        data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        data, labels = self.preprocess(data, config)
        return data, labels

    def get_data(self):
        config = self.load_config()
        data, labels = self.load_data(config)
        X_train, X_test, y_train, y_test = self.split_data(data, labels)
        dataset = [X_train, X_test, y_train, y_test, config]

        return dataset
        

        



