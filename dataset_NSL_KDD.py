import pandas as pd
import os
import json
from scipy.io import arff

class Dataset_NSL_KDD:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, config_filename=None):
        self.CONFIG_FILE_PATH = f"{self.BASE_DIR}/config/{config_filename}"
        self.config = self.load_config()
        self.filename = self.config['name_dataset']
        self.DATA_FILE_PATH = f"{self.BASE_DIR}/data/NSL-KDD/{self.filename}"
        self.data = None
        self.labels = None
        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def load_data(self):
        features=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
        
        # Load data from CSV file
        df = pd.read_csv(self.DATA_FILE_PATH, sep=",", header=None, names=features, nrows=self.config['nrows'])
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

        self.labels = df['label'] # Pandas Series 
        self.data = df.drop('label', axis=1) # Pandas DataFrame
