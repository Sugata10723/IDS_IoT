import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing
import plotter 
import matplotlib.pyplot as plt
from scipy import stats

#################################################################################   
# パラメータ
# k: クラスタ数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの外れ値の割合
#################################################################################


class AnomalyDetector_var:
    def __init__(self, parameters, categorical_columns):
        self.k = parameters['k']
        self.n_estimators = parameters['n_estimators']
        self.max_features = parameters['max_features']
        self.n_ohe = parameters['n_ohe']
        self.n_num = parameters['n_num']
        self.c_attack = parameters['c_attack']
        self.c_normal = parameters['c_normal']
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=500, max_features=self.max_features, contamination=self.c_attack)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=500, max_features=self.max_features, contamination=self.c_normal)

        self.features_num_attack = None
        self.features_ohe_attack = None
        self.features_num_normal = None
        self.features_ohe_normal = None
        self.sampled_attack = None
        self.sampled_normal = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None
        self.attack_prd = None
        self.normal_prd = None
        self.X_attack = None
        self.X_normal = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]

    def feature_selection(self, X_ohe, X_num):
        #最初に分散が0のデータを消す
        X_num = X_num[:, X_num.var(axis=0) != 0]
        X_ohe = X_ohe[:, X_ohe.var(axis=0) != 0]
        # 数値データに対して分散を計算し、小さい特徴量を順にself.n_num個選択
        var_num = np.var(X_num, axis=0)
        features_num = np.argsort(var_num)[:self.n_num]
        # カテゴリデータに対しても同様
        var_ohe = np.var(X_ohe, axis=0)
        features_ohe = np.argsort(var_ohe)[:self.n_ohe]
        # 選択された特徴量を用いてデータを結合
        data_fi = np.concatenate([X_ohe[:, features_ohe], X_num[:, features_num]], axis=1)

        return data_fi, features_num, features_ohe

    def get_nearest_points(self, data, kmeans):
        distances = kmeans.transform(data[:, :-1])
        nearest_points = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(data[:, -1] == i)[0]
            if len(cluster_indices) > 0:  # k-meansの特性より、空のクラスタが存在する可能性がある
                nearest_index = cluster_indices[np.argmin(distances[cluster_indices, i])]
                nearest_points.append(data[nearest_index])
        nearest_points = np.array(nearest_points)
        # サンプリング数がクラスタ数より少ない場合は、足りない分をランダムサンプリング
        if len(nearest_points) < self.k:
            random_indices = np.random.choice(len(data), self.k - len(nearest_points), replace=False)
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
            print(f"sampled data is :{data_sampled.shape}")
            return data_sampled

    def fit(self, X, y):
        ## Preprocessing X: Pandas DataFrame, y: NumPy Array
        # one-hotエンコード 入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.fit_transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        # 正規化 入力：ndarray　出力：ndarray
        X_num = self.mm.fit_transform(X_num)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        X_ohe_attack, X_ohe_normal = self.splitsubsystem(X_ohe, y)
        X_num_attack, X_num_normal = self.splitsubsystem(X_num, y)
        # 特徴量選択 入力：ndarray 出力：ndarray
        self.attack_data, self.features_num_attack, self.features_ohe_attack = self.feature_selection(X_ohe_attack, X_num_attack)
        self.normal_data, self.features_num_normal, self.features_ohe_normal = self.feature_selection(X_ohe_normal, X_num_normal)
        # サンプリング 入力：ndarray　出力：ndarray
        self.sampled_attack = self.make_cluster(self.attack_data)
        self.sampled_normal = self.make_cluster(self.normal_data)
    
        ## training 入力：ndarray
        self.iforest_attack.fit(self.sampled_attack)
        self.iforest_normal.fit(self.sampled_normal)

        
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
        # 特徴量選択 入力：ndarray 出力：ndarray
        self.X_attack = np.concatenate([X_ohe[:, self.features_ohe_attack], X_num[:, self.features_num_attack]], axis=1)
        self.X_normal = np.concatenate([X_ohe[:, self.features_ohe_normal], X_num[:, self.features_num_normal]], axis=1)

        ## predict
        attack_prd = self.iforest_attack.predict(self.X_attack)
        attack_prd = [1 if result == 1 else 0 for result in attack_prd]   
        self.attack_prd = attack_prd
        
        normal_prd = self.iforest_normal.predict(self.X_normal)
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
        scores_attack = self.iforest_attack.decision_function(self.X_attack)
        scores_normal = self.iforest_normal.decision_function(self.X_normal)
        plt.hist(scores_attack, bins=50, alpha=0.5, label='attack')
        plt.hist(scores_normal, bins=50, alpha=0.5, label='normal')
        plt.legend()
        plt.show()
