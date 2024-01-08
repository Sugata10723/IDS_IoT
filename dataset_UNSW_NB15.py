import pandas as pd
import os
import json
########################################
# Dataset_UNSW_NB15
# 最終的な変換後の特徴量数は72
# dsportはobject型であり、16進数と10進数が混在するため、10進数に統一する
# sportはint64型であり、10進数で記録されているため、そのまま使用する
# dstip, srcipはobject型であり、IPアドレスで記録されているため、4分割して整数値(int64)に変換する
########################################
class Dataset_UNSW_NB15:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, config_filename='config.json'):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', config_filename)
        self.config = self.load_config()
        self.filename = self.config['name_dataset']
        self.DATA_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/{self.filename}"
        self.DATA_FEATURES_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/test.csv"
        self.data = None
        self.labels = None
        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def convert_to_decimal(self, x):
        try:
            # 16進数として解釈して変換
            return int(x, 16)
        except ValueError:
            # 10進数として解釈
            return int(x)

    def split_ip(self, ip):
        return list(map(int, ip.split('.')))

    def load_data(self):
        nrows = int(self.config['nrows'])
        self.data = pd.read_csv(self.DATA_FILE_PATH, nrows=nrows)
        features = pd.read_csv(self.DATA_FEATURES_FILE_PATH, index_col='No.')
        features = features['Name']
        self.data.columns = features

        # 不要なカラムを削除
        self.data.drop(columns=self.config['unwanted_columns'], inplace=True)

        # dsportを変換
        self.data['dsport'] = self.data['dsport'].apply(self.convert_to_decimal)

        # dstip, srcipを整数値に変換する
        self.data['dstip_1'], self.data['dstip_2'], self.data['dstip_3'], self.data['dstip_4'] = zip(*self.data['dstip'].apply(self.split_ip))
        self.data.drop(columns=['dstip'], inplace=True)
        self.data['srcip_1'], self.data['srcip_2'], self.data['srcip_3'], self.data['srcip_4'] = zip(*self.data['srcip'].apply(self.split_ip))
        self.data.drop(columns=['srcip'], inplace=True)

        self.labels = self.data['Label'] # Pandas Series
        self.data = self.data.drop(columns=['Label']) # Pandas DataFrame

        



