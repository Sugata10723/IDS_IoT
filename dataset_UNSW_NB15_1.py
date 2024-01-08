import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

########################################
# Dataset_UNSW_NB15_1
# 元がAttackが少ないデータセット
# 元の特徴量は49個
# 最終的な変換後の特徴量数は72
# 1, 3, 47行目に異なるデータ型が混在している
# dsportはobject型であり、16進数と10進数が混在するため、10進数に統一する。異常な値は削除する
# sportはint64型であり、10進数で記録されているため、そのまま使用する。47312, 116466に文字列が混在するので、削除する
# dstip, srcipはobject型であり、IPアドレスで記録されているため、4分割して整数値(int64)に変換する
########################################
class Dataset_UNSW_NB15_1:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, nrows):
        self.nrows = nrows
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', 'config_UNSW_NB15_1.json')
        self.config = self.load_config()
        self.DATA_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/UNSW-NB15_1.csv"
        self.DATA_FEATURES_FILE_PATH = f"{self.BASE_DIR}/data/UNSW_NB15/test.csv"
        self.data = None
        self.labels = None
        self.X_train = None # Pandas DataFrame
        self.X_test = None # Pandas DataFrame
        self.y_train = None # Pandas Series
        self.y_test = None # Pandas Series

        self.load_data()
        self.split_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def convert_to_decimal(self, x):
        if not isinstance(x, str):
            return x
        try:
            # 16進数として解釈して変換
            return int(x, 16)
        except ValueError:
            try:
                # 10進数として解釈
                return int(x)
            except ValueError:
                # 無効な値が渡された場合は-1を返す
                return -1

    def split_ip(self, ip):
        return list(map(int, ip.split('.')))

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )
        self.y_train = self.y_train.values # Pandas Series -> NumPy Array
        self.y_test = self.y_test.values # Pandas Series -> NumPy Array

    def load_data(self):
        self.data = pd.read_csv(self.DATA_FILE_PATH)
        features = pd.read_csv(self.DATA_FEATURES_FILE_PATH, index_col='No.')
        features = features['Name']
        self.data.columns = features

        # 指定した行数だけ読み込む
        if self.nrows > self.data.shape[0]:
            self.nrows = self.data.shape[0]
        self.data = self.data.iloc[:self.nrows]

        # 不要なカラムを削除
        self.data.drop(columns=self.config['unwanted_columns'], inplace=True)

        # dsportを変換
        self.data['dsport'] = self.data['dsport'].apply(self.convert_to_decimal)
        # dsportの値が-1の行（すなわち、無効な値を含む行）を削除し、その数をカウント
        n_invalid_dsport = len(self.data[self.data['dsport'] == -1])
        self.data = self.data[self.data['dsport'] != -1]
        print(f'Invalid dsport: {n_invalid_dsport}')

        # sportに含まれる文字列を削除
        is_int = self.data['sport'].apply(lambda x: isinstance(x, int))
        self.data = self.data[is_int]

        # dstip, srcipを整数値に変換する
        self.data['dstip_1'], self.data['dstip_2'], self.data['dstip_3'], self.data['dstip_4'] = zip(*self.data['dstip'].apply(self.split_ip))
        self.data.drop(columns=['dstip'], inplace=True)
        self.data['srcip_1'], self.data['srcip_2'], self.data['srcip_3'], self.data['srcip_4'] = zip(*self.data['srcip'].apply(self.split_ip))
        self.data.drop(columns=['srcip'], inplace=True)

        self.data = self.data.reset_index() 
        self.labels = self.data['Label'] # Pandas Series
        self.data = self.data.drop(columns=['Label']) # Pandas DataFrame

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config

        



