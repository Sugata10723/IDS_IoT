import pandas as pd
from zipfile import ZipFile
import os
import json
import numpy as np
from sklearn.impute import SimpleImputer

#################################################################################
# Dataset_IoT_Network_Intrusion_Dataset
# 最終的な変換後の特徴量数は
# Src_IPとDst_IPを整数値に変換
# infがある行があるので削除する
#################################################################################

class Dataset_IoT_Network_Intrusion_Dataset:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, config_filename='config.json'):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', config_filename)
        self.config = self.load_config()
        self.filename = self.config['name_dataset']

        self.DATA_CSV_FILE_PATH = f"{self.BASE_DIR}/data/{self.filename}/{self.filename}.csv"
        
        self.data = None
        self.labels = None
        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def split_ip(self, ip):
        return list(map(int, ip.split('.')))

    def load_data(self):
        self.data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        nrows = self.config['nrows']
        anomaly_data = self.data.loc[self.data['Label'] == 'Anomaly'].head(int(nrows/2)).copy()
        normal_data = self.data.loc[self.data['Label'] == 'Normal'].head(int(nrows/2)).copy()

        self.data = pd.concat([anomaly_data, normal_data])

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
        

        



