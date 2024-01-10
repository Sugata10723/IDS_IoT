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

    def __init__(self, nrows):
        self.nrows = nrows
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

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
        self.y_train = self.y_train.values # Pandas Series -> NumPy Array
        self.y_test = self.y_test.values # Pandas Series -> NumPy Array

    def bitwise(self, data):
        data = data.copy()

        # IPアドレスを32ビットのバイナリ文字列に変換する関数
        def ip_to_bin(ip):
            return ''.join([format(int(x), '08b') for x in ip.split('.')])

        # dstipとsrcipを32ビットのバイナリ文字列に変換
        data['dstip_bin'] = data['Dst_IP'].apply(ip_to_bin)
        data['srcip_bin'] = data['Src_IP'].apply(ip_to_bin)

        # バイナリ文字列を個々のビットに分割し、新しい特徴量として追加
        for i in range(32):
            data[f'dstip_bit_{i}'] = data['dstip_bin'].apply(lambda x: int(x[i]))
            data[f'srcip_bit_{i}'] = data['srcip_bin'].apply(lambda x: int(x[i]))

        # バイナリ文字列の特徴量はもう不要なので削除
        data.drop(columns=['dstip_bin', 'srcip_bin', 'Dst_IP', 'Src_IP'], inplace=True)

        return data

    def cut_data(self, data):
        data = data.copy()
        nrows = min(self.nrows, data.shape[0]) # 行数がnrowsよりも少ない場合は、dataをそのまま返す
        if self.config['fix_imbalance']: # 不均衡データを均衡データにする場合
            data_0 = data[data['Label'] == 0]
            data_1 = data[data['Label'] == 1]
            half_nrows = nrows // 2
            data = pd.concat([data_0.iloc[:half_nrows], data_1.iloc[:half_nrows]])
        else: # 不均衡データのまま
            data = data.iloc[:nrows]
        return data

    def load_data(self):
        self.data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        self.data['Label'] = self.data['Label'].map({'Anomaly': 1, 'Normal': 0}) # Pandas Series

        # 指定した行数だけ読み込む
        self.data = self.cut_data(self.data)

        #　不用なカラムを削除
        self.data.drop(columns=self.config["unwanted_columns"], inplace=True)
        
        # infのある行を削除
        inf_rows = self.data.isin([np.inf, -np.inf]).any(axis=1)
        if inf_rows.sum() > 0:
            print(f"Number of rows with inf: {inf_rows.sum()}")
        self.data = self.data[~inf_rows]
        self.data.reset_index(drop=True, inplace=True)

        # IPアドレスを整数値に変換
        self.data = self.bitwise(self.data)

        self.labels = self.data['Label'] # Pandas Series
        self.data.drop(columns=['Label'], inplace=True) # Pandas DataFrame

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config
        

        



