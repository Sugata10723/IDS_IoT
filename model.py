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
        self.important_features_attack = None
        self.important_features_normal = None
        self.pca_attack = None
        self.pca_normal = None
        self.sampled_attack = None
        self.sampled_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        self.max_feature = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]

    def feature_selection(self, data):
        # 特徴量の分散を計算
        variances = np.var(data, axis=0)
        # 分散の分布を表示
        plt.hist(variances, bins='auto')
        plt.title('Distribution of Feature Variances')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.show()
        # 分散が0.05を超える特徴量はカットする
        important_features = np.where(variances < 0.05)[0]
        data_fi = data[:, important_features]
        data_fs = data_fi 

        # 重要な特徴量を除いた特徴量からPCAで特徴抽出
        #remaining_features = np.delete(np.arange(data.shape[1]), important_features)
        #remaining_data = data[:, remaining_features]
        pca = PCA(n_components=self.n_pca)
        #data_pca = pca.fit_transform(remaining_data)
        #data_fs = np.concatenate([data_pca, data_fi], axis=1) # debagのため

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
        X = np.concatenate([X.drop(columns=self.categorical_columns).values, X_ohe], axis=1)
        self.max_feature = X.shape[1] # プロットのために記録
        print(f'max_feature: {self.max_feature}')
        # 正規化 入力：ndarray　出力：ndarray
        X = self.mm.fit_transform(X)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        raw_attack_data, raw_ormal_data = self.splitsubsystem(X, y)
        # 特徴量選択 入力：ndarray 出力：ndarray
        self.attack_data, self.pca_attack, self.important_features_attack = self.feature_selection(raw_attack_data)
        self.normal_data, self.pca_normal, self.important_features_normal = self.feature_selection(raw_ormal_data)
        # クラスタリング 入力：ndarray　出力：ndarray
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
        X = np.concatenate([X.drop(columns=self.categorical_columns).values, X_ohe], axis=1)
        # 正規化　入力：ndarray　出力：ndarray
        X = self.mm.transform(X)
        # 特徴量選択 入力：ndarray 出力：ndarray
        X_attack = X[:, self.important_features_attack]
        X_normal = X[:, self.important_features_normal]
        # 特徴抽出 入力：ndarray　出力：ndarray        

        ## predict
        attack_results = self.iforest_attack.predict(X_attack)
        attack_results = [1 if result == 1 else 0 for result in attack_results]   
        
        normal_results = self.iforest_normal.predict(X_normal)
        normal_results = [1 if result == 1 else 0 for result in normal_results]
        
        for i in range(total_points):
            if attack_results[i] == 0 and normal_results[i] == 1: # normal
                predictions.append(0)  
            elif attack_results[i] == 0 and normal_results[i] == 0: # unknown
                predictions.append(-1)  
            else: # attack
                predictions.append(1)

        return predictions
