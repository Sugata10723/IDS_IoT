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

class Dataset_IoT_Network_Intrusion_Dataset:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_IoT_Network_Intrusion_Dataset.json')
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

    def split_ip(self, ip):
        return list(map(int, ip.split('.')))

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
        self.y_train = self.y_train.values # Pandas Series -> NumPy Array
        self.y_test = self.y_test.values # Pandas Series -> NumPy Array

    def load_data(self):
        self.data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        nrows = self.config['nrows']
        self.data = self.data.head(nrows)

        #　不用なカラムを削除
        self.data.drop(columns=self.config["unwanted_columns"], inplace=True)
        
        # infのある行を削除
        inf_rows = self.data.isin([np.inf, -np.inf]).any(axis=1)
        if inf_rows.sum() > 0:
            print(f"Number of rows with inf: {inf_rows.sum()}")
        self.data = self.data[~inf_rows]
        self.data.reset_index(drop=True, inplace=True)

        # Src_IPとDst_IPを整数値に変換
        self.data['Src_IP_1'], self.data['Src_IP_2'], self.data['Src_IP_3'], self.data['Src_IP_4'] = zip(*self.data['Src_IP'].apply(self.split_ip))
        self.data.drop(columns=['Src_IP'], inplace=True)
        self.data['Dst_IP_1'], self.data['Dst_IP_2'], self.data['Dst_IP_3'], self.data['Dst_IP_4'] = zip(*self.data['Dst_IP'].apply(self.split_ip))
        self.data.drop(columns=['Dst_IP'], inplace=True)

        self.labels = self.data['Label'].map({'Anomaly': 1, 'Normal': 0}) # Pandas Series
        self.data.drop(columns=['Label'], inplace=True) # Pandas DataFrame

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config
        

        



