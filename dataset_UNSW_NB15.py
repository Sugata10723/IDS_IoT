import pandas as pd
import os
import json

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

    def load_data(self):
        nrows = int(self.config['nrows'])
        self.data = pd.read_csv(self.DATA_FILE_PATH, nrows=nrows)
        features = pd.read_csv(self.DATA_FEATURES_FILE_PATH, index_col='No.')
        features = features['Name']
        self.data.columns = features

        # 不要なカラムを削除
        self.data = self.data.drop(columns=self.config['unwanted_columns'])

        # sport, dsportの一部が文字データになっているので削除する
        #for col in ['sport', 'dsport']:
        #    if col in self.data.columns:
        #        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        #self.data = self.data.dropna(subset=['sport', 'dsport'])
        #self.data.reset_index(drop=True, inplace=True) # indexを振り直す

        self.labels = self.data['Label'] # Pandas Series
        self.data = self.data.drop(columns=['Label']) # Pandas DataFrame

        



