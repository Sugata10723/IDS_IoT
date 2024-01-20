import pandas as pd
import os
import json
from scipy.io import arff

#################################################################################
# Dataset_NSL_KDD
# trainデータ数: 125973
# testデータ数: 22544
# 特徴量数: 41
# 最終的な変換後の特徴量数は
# duration, protocol_type, service, flagはカテゴリカル変数であり、One-Hotエンコーディングを行う
# difficultyは不要なカラムとして削除する
#################################################################################

class Dataset_NSL_KDD:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.CONFIG_FILE_PATH = f"{self.BASE_DIR}/config/config_NSL_KDD.json"
        self.config = self.load_config()
        self.DATA_TRAIN_FILE_PATH = f"{self.BASE_DIR}/data/NSL-KDD/KDDTrain+.txt"
        self.DATA_TEST_FILE_PATH = f"{self.BASE_DIR}/data/NSL-KDD/KDDTest+.txt"
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.load_data()

    def load_config(self):
        with open(self.CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)

    def preprocess(self, data):
        # グローバル変数を変更しないようにコピー
        data = data.copy()
        # labelを正常:0, 攻撃:1に変換
        data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
        # 必要ない列を削除
        data = data.drop(columns=self.config['unwanted_columns'])
        # ラベルを分割
        labels = data['label'] # Pandas Series 
        data = data.drop('label', axis=1) # Pandas DataFrame

        return data, labels

    def load_data(self):
        features=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
        
        # Load data from CSV file
        self.X_train = pd.read_csv(self.DATA_TRAIN_FILE_PATH, sep=",", header=None, names=features)
        self.X_test = pd.read_csv(self.DATA_TEST_FILE_PATH, sep=",", header=None, names=features)

        # Preprocess data
        self.X_train, self.y_train = self.preprocess(self.X_train)
        self.X_test, self.y_test = self.preprocess(self.X_test)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.config
