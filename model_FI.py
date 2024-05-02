import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing
import plotter 
import matplotlib.pyplot as plt

#################################################################################   
# パラメータ
# k: クラスタ数
# n_fi: 特徴量選択における重要な特徴量の数
# n_pca: 特徴量選択におけるPCAによる特徴抽出の次元数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの外れ値の割合
#################################################################################


class AnomalyDetector_FI:
    def __init__(self, parameters, categorical_columns):
        self.k = parameters['k']
        self.n_estimators = parameters['n_estimators']
        self.max_features = parameters['max_features']
        self.max_samples = parameters['max_samples']
        self.c_attack =  parameters['c_attack']
        self.c_normal = parameters['c_normal']
        self.threshold = parameters['threshold']
        self.categorical_columns = categorical_columns

        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_attack)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_normal)

        self.important_features = None
        self.sampled_attack = None
        self.sampled_normal = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None
        self.normal_prd = None
        self.attack_prd = None
        self.X_processed = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]

    def feature_selection(self, data, y):
        model = RandomForestClassifier() # 使用するモデルはrandomforest
        model.fit(data, y)
        feature_importances = model.feature_importances_
        important_features = np.where(feature_importances > self.threshold)[0]
        data_fi = data[:, important_features]
        self.plot_feature_importance(feature_importances)

        return data_fi, important_features

    def plot_feature_importance(self, feature_importances):
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.show()

    def fit(self, X, y):
        ## Preprocessing X: Pandas DataFrame, y: NumPy Array
        # one-hotエンコード 入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.fit_transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        # 正規化 入力：ndarray　出力：ndarray
        X_num = self.mm.fit_transform(X_num)
        X_processed = np.concatenate([X_ohe, X_num], axis=1)
        # 特徴量選択 入力：ndarray 出力: ndarray
        X_fs, self.important_features = self.feature_selection(X_processed, y)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        self.attack_data, self.normal_data = self.splitsubsystem(X_fs, y)
    
        ## training 入力：ndarray
        self.iforest_attack.fit(self.attack_data)
        self.iforest_normal.fit(self.normal_data)

        
    def predict(self, X):
        predictions = []
        total_points = len(X)

        ## preprocessing
        # one-hotエンコード　入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        # 正規化　入力：ndarray 出力：ndarray
        X_num = self.mm.transform(X_num) # numeicalデータにだけでよい
        X_processed = np.concatenate([X_ohe, X_num], axis=1)
        # 特徴量選択 入力：ndarray 出力：ndarray    
        self.X_processed = X_processed[:, self.important_features]
    
        ## predict
        attack_prd = self.iforest_attack.predict(self.X_processed)
        attack_prd = [1 if result == 1 else 0 for result in attack_prd]   
        self.attack_prd = attack_prd
        
        normal_prd = self.iforest_normal.predict(self.X_processed)
        normal_prd = [1 if result == 1 else 0 for result in normal_prd]
        self.normal_prd = [0 if x == 1 else 1 for x in normal_prd] # normalの判定は逆になる
        
        for i in range(total_points):
            if attack_prd[i] == 0 and normal_prd[i] == 1: # normal
                predictions.append(0)  
            elif attack_prd[i] == 0 and normal_prd[i] == 0: # unknown
                predictions.append(-1)  
            else: # attack
                predictions.append(1)

        return predictions

    def plot_anomaly_scores(self):
        scores_attack = self.iforest_attack.decision_function(self.X_processed)
        scores_normal = self.iforest_normal.decision_function(self.X_processed)
        plt.hist(scores_attack, bins=50, alpha=0.5, label='attack')
        plt.hist(scores_normal, bins=50, alpha=0.5, label='normal')
        plt.legend()
        plt.show()
