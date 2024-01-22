import pandas as pd
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.decomposition import PCA
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing
import plotter 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#################################################################################   
# パラメータ
# k: クラスタ数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの外れ値の割合
#################################################################################


class AnomalyDetector_var:
    def __init__(self, k=1, categorical_columns=None):
        self.k = k
        self.n_estimators = 50
        self.max_samples = 100
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.important_features_attack = None
        self.important_features_normal = None
        self.sampled_attack = None
        self.sampled_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None
        self.attack_prd = None
        self.normal_prd = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]

    def feature_selection(self, data):
        # 特徴量の分散を計算
        variances = np.var(data, axis=0)
        threshold = np.median(variances)
        # 分散の分布を表示
        plt.hist(variances, bins=50)
        plt.title(f'Distribution of Feature Variances: median={round(threshold, 4)}')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.show()
        # 分散の中央値以上の分散を持つ特徴量を削除
        important_features = np.where(variances > threshold)[0]
        data_fi = data[:, important_features]

        return data_fi, important_features

    def get_nearest_points(self, data, kmeans):
        distances = kmeans.transform(data[:, :-1])
        nearest_points = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(data[:, -1] == i)[0]
            if len(cluster_indices) > 0:  # k-meansの特性より、空のクラスタが存在する可能性がある
                nearest_index = cluster_indices[np.argmin(distances[cluster_indices, i])]
                nearest_points.append(data[nearest_index])
        nearest_points = np.array(nearest_points)
        nearest_points = nearest_points[:, :-1] 
        return nearest_points

    def make_cluster(self, data):
        if len(data) < self.k: # サンプル数がkより小さい場合はそのまま返す
            return data
        else:
            start_time = time.time()
            kmeans = MiniBatchKMeans(n_clusters=self.k, init='k-means++', batch_size=100, tol=0.01, n_init=10) 
            clusters = kmeans.fit_predict(data)
            print(f"K-means clustering time: {round(time.time() - start_time, 3)}sec")
            data = np.column_stack((data, clusters)) 
            data_sampled = self.get_nearest_points(data, kmeans)
            print(f"Sampling time: {round(time.time() - finish_kmeans, 3)}sec")
            return data_sampled

    def fit(self, X, y):
        ## Preprocessing X: Pandas DataFrame, y: NumPy Array
        # one-hotエンコード 入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.fit_transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        print(f"X_ohe shape is: {X_ohe.shape[1]}")
        print(f"X_num shape is: {X_num.shape[1]}")
        # 正規化 入力：ndarray　出力：ndarray
        X_num = self.mm.fit_transform(X_num)
        X_processed = np.concatenate([X_num, X_ohe], axis=1)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        raw_attack_data, raw_normal_data = self.splitsubsystem(X_processed, y)
        # 特徴量選択 入力：ndarray 出力：ndarray
        self.attack_data, self.important_features_attack = self.feature_selection(raw_attack_data)
        self.normal_data, self.important_features_normal = self.feature_selection(raw_normal_data)
        # サンプリング 入力：ndarray　出力：ndarray
        self.sampled_attack = self.make_cluster(self.attack_data)
        self.sampled_normal = self.make_cluster(self.normal_data)
    
        ## training 入力：ndarray
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=min(self.max_samples, len(self.sampled_attack))).fit(self.sampled_attack)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=min(self.max_samples, len(self.sampled_normal))).fit(self.sampled_normal)

        
    def predict(self, X):
        attack_results = []
        normal_results = []
        predictions = []
        total_points = len(X)

        ## preprocessing
        # one-hotエンコード　入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        # 正規化　入力：ndarray　出力：ndarray
        X_num = self.mm.transform(X_num)
        X_processed = np.concatenate([X_num, X_ohe], axis=1)
        # 特徴量選択 入力：ndarray 出力：ndarray
        X_attack = X_processed[:, self.important_features_attack]
        X_normal = X_processed[:, self.important_features_normal]     

        ## predict
        attack_prd = self.iforest_attack.predict(X_attack)
        attack_prd = [1 if result == 1 else 0 for result in attack_prd]   
        self.attack_prd = attack_prd
        
        normal_prd = self.iforest_normal.predict(X_normal)
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
