import pandas as pd
import os
import json

class Dataset_UNSW_NB15:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, config_filename='config.json'):
        self.CONFIG_FILE_PATH = os.path.join(self.BASE_DIR, 'config', config_filename)
        self.config = self.load_config()
        self.filename = self.config['name_dataset']

        self.DATA_TRAINING_CSV_FILE_PATH = f"{self.BASE_DIR}/data/{self.filename}/{self.filename}_training-set.csv"
        self.DATA_TESTING_CSV_FILE_PATH = f"{self.BASE_DIR}/data/{self.filename}/{self.filename}_testing-set.csv"
        
        self.training_data = None
        self.testing_data = None
        self.labels = None
        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def load_data(self):
        self.training_data = pd.read_csv(self.DATA_TRAINING_CSV_FILE_PATH, nrows=self.config['num_rows'])
        self.testing_data = pd.read_csv(self.DATA_TESTING_CSV_FILE_PATH, nrows=self.config['num_rows'])
        
        self.data = pd.concat([self.training_data, self.testing_data])
        self.labels = self.data['label'].map({1: 1, 0: 0})
        
        self.data = pd.DataFrame(self.data)
        self.labels = pd.DataFrame(self.labels)

        



