import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as preprocessing
from sklearn_extra.cluster import KMedoids


#################################################################################   
# AnomalyDetector_noFS
# feature selectionなし
# パラメータ
# k: クラスタ数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの異常スコアの閾値
#################################################################################


class AnomalyDetector_noFS:
    def __init__(self, k, c_attack, c_normal, categorical_columns=None):
        self.k = k
        self.c_attack = c_attack
        self.c_normal = c_normal
        self.n_estimators = 100
        self.max_samples = 500
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
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

    def get_nearest_points(self, data, kmeans):
        distances = kmeans.transform(data[:, :-1])
        nearest_points = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(data[:, -1] == i)[0]
            if len(cluster_indices) > 0:  # k-meansの特性より、空のクラスタが存在する可能性がある
                nearest_index = cluster_indices[np.argmin(distances[cluster_indices, i])]
                nearest_points.append(data[nearest_index])
        nearest_points = np.array(nearest_points)
        if len(nearest_points) < self.k:  # サンプル数がkより小さい場合は残りをランダムサンプリングで埋める
            random_indices = np.random.choice(data.shape[0], self.k - len(nearest_points), replace=False)
            nearest_points = np.concatenate([nearest_points, data[random_indices]])
        nearest_points = nearest_points[:, :-1]
        return nearest_points

    def make_cluster(self, data):
        if len(data) < self.k: # サンプル数がkより小さい場合はそのまま返す
            return data
        else:
            kmeans = MiniBatchKMeans(n_clusters=self.k, init='k-means++', batch_size=100, tol=0.01, n_init=10) 
            clusters = kmeans.fit_predict(data)
            data = np.column_stack((data, clusters))
            data_sampled = self.get_nearest_points(data, kmeans)
            print(f"sampled Data shape is: {data_sampled.shape}")
            return data_sampled

    def preprocess(self, X, if_train):
        # one-hotエンコード 入力：DataFrame　出力：ndarray
        if if_train==True:
            self.ohe.fit(X[self.categorical_columns])
        X_ohe = self.ohe.transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns).values
        # 正規化 入力：ndarray　出力：ndarray
        X_num = self.mm.fit_transform(X_num)
        X_processed = np.concatenate([X_num, X_ohe], axis=1)
        # 特徴量数の表示 
        print(f"X_ohe shape is: {X_ohe.shape[1]}")
        print(f"X_num shape is: {X_num.shape[1]}")
        return X_processed
            
    def fit(self, X, y):
        ## 前処理 入力：DataFrame 出力：ndarray
        X_processed = self.preprocess(X, if_train=True)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        self.attack_data, self.normal_data = self.splitsubsystem(X_processed, y)
        # サンプリング 入力：ndarray　出力：ndarray
        self.sampled_attack = self.make_cluster(self.attack_data)
        self.sampled_normal = self.make_cluster(self.normal_data)
        print(f"sampled attack shape is: {self.sampled_attack.shape}")
        print(f"sampled normal shape is: {self.sampled_normal.shape}")

    
        # トレーニング 入力：ndarray
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,  max_features=5, contamination=self.c_attack).fit(self.sampled_attack)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,  max_features=5, contamination=self.c_normal).fit(self.sampled_normal)
        
    def predict(self, X):
        attack_results = []
        normal_results = []
        predictions = []
        total_points = len(X)

        # 前処理
        X_processed = self.preprocess(X, if_train=False)
        # 予測
        attack_prd = self.iforest_attack.predict(X_processed)
        attack_prd = [1 if result == 1 else 0 for result in attack_prd]   
        self.attack_prd = attack_prd
        
        normal_prd = self.iforest_normal.predict(X_processed)
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
