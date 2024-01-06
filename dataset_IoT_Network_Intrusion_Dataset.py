import pandas as pd
from zipfile import ZipFile
import os
import json

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

    def load_data(self):
        self.data = pd.read_csv(self.DATA_CSV_FILE_PATH)
        nrows = self.config['nrows']
        anomaly_data = self.data.loc[self.data['Label'] == 'Anomaly'].head(int(nrows/2)).copy()
        normal_data = self.data.loc[self.data['Label'] == 'Normal'].head(int(nrows/2)).copy()

        self.data = pd.concat([anomaly_data, normal_data])
        self.labels = self.data['Label'].map({'Anomaly': 1, 'Normal': 0}) # Pandas Series
        self.data.drop(columns=['Label'], inplace=True) # Pandas DataFrame
        

        



