import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
# n_fi: 特徴量選択における重要な特徴量の数
# n_pca: 特徴量選択におけるPCAによる特徴抽出の次元数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの外れ値の割合
#################################################################################


class AnomalyDetector:
    def __init__(self, k=1, n_fi=1, n_pca=1, categorical_columns=None):
        self.k = k
        self.n_estimators = 50
        self.max_samples = 100
        self.n_fi = n_fi
        self.n_pca = n_pca
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.important_features = None
        self.pca = None
        self.sampled_attack = None
        self.sampled_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]

    def feature_selection(self, categorical_data, numerical_data, y):
        # カテゴリ変数に対してFIを用いて特徴量選択
        xgb_model = xgb.XGBClassifier() # 使用するモデルは要検討
        xgb_model.fit(categorical_data, y)
        feature_importances = xgb_model.feature_importances_
        important_features = np.argsort(feature_importances)[::-1][:self.n_fi]
        data_fi = categorical_data[:, important_features]
        # 数値変数に対してPCAを用いて特徴量選択
        pca = PCA(n_components=self.n_pca)
        data_pca = pca.fit_transform(numerical_data)
        # 特徴量選択後のデータを結合 出力：DataFrame
        data_fs = np.concatenate([data_fi, data_pca], axis=1)

        return data_fs, pca, important_features 

    def get_nearest_points(self, data, kmeans):
        nearest_points = []
        for i, center in enumerate(kmeans.cluster_centers_):
            cluster_data = data[data[:, -1] == i]
            if len(cluster_data) > 0:  # k-meansの特性より、空のクラスタが存在する可能性がある
                distances = np.apply_along_axis(lambda x: distance.euclidean(x[:-1], center), 1, cluster_data)
                nearest_point = np.argmin(distances)
                nearest_points.append(cluster_data[nearest_point])
        nearest_points = np.array(nearest_points)
        nearest_points = nearest_points[:, :-1]  # 'cluster' columnを削除
        return nearest_points

    def make_cluster(self, data):
        if len(data) < self.k: # サンプル数がkより小さい場合はそのまま返す
            return data
        else:
            kmeans = KMeans(n_clusters=self.k, n_init=10)
            clusters = kmeans.fit_predict(data)
            data = np.column_stack((data, clusters))  # 'cluster' columnを追加
            data_sampled = self.get_nearest_points(data, kmeans)
            return data_sampled

    def fit(self, X, y):
        ## Preprocessing X: Pandas DataFrame, y: NumPy Array
        # one-hotエンコード 入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.fit_transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns).values
        print(f"X_num shape is: {X_num.shape}")
        print(f"X_ohe shape is: {X_ohe.shape}")
        # 正規化 入力：ndarray　出力：ndarray
        X_num = self.mm.fit_transform(X_num)
        # 特徴量選択 入力：ndarray 出力: ndarray
        X_fs, self.pca, self.important_features = self.feature_selection(X_ohe, X_num, y)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        self.attack_data, self.normal_data = self.splitsubsystem(X_fs, y)
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
        X_num = X.drop(columns=self.categorical_columns).values
        # 正規化　入力：ndarray 出力：ndarray
        X_num = self.mm.transform(X_num) # numeicalデータにだけでよい
        # 特徴量選択 入力：ndarray 出力：ndarray    
        X_ohe = X_ohe[:, self.important_features]
        X_num = self.pca.transform(X_num)
        X = np.concatenate([X_ohe, X_num], axis=1)
    
        ## predict
        attack_results = self.iforest_attack.predict(X)
        attack_results = [1 if result == 1 else 0 for result in attack_results]   
        
        normal_results = self.iforest_normal.predict(X)
        normal_results = [1 if result == 1 else 0 for result in normal_results]
        
        for i in range(total_points):
            if attack_results[i] == 0 and normal_results[i] == 1: # normal
                predictions.append(0)  
            elif attack_results[i] == 0 and normal_results[i] == 0: # unknown
                predictions.append(-1)  
            else: # attack
                predictions.append(1)

        return predictions
