import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.decomposition import PCA
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


class AnomalyDetector_hybrid:
    def __init__(self, parameters, categorical_columns):
        self.k = parameters['k']
        self.n_estimators = parameters['n_estimators']
        self.max_features = parameters['max_features']
        self.max_samples = parameters['max_samples']
        self.c_attack =  parameters['c_attack']
        self.c_normal = parameters['c_normal']
        self.n_fi = parameters['n_fi']
        self.n_pca = parameters['n_pca']
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_attack)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_normal)
        self.pca = PCA(n_components=self.n_pca)

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

    def feature_selection(self, categorical_data, numerical_data, y):
        # カテゴリ変数に対してFIを用いて特徴量選択
        model = RandomForestClassifier() # 使用するモデルはrandomforest
        model.fit(categorical_data, y)
        feature_importances = model.feature_importances_
        important_features = np.argsort(feature_importances)[::-1][:self.n_fi]
        data_fi = categorical_data[:, important_features]
        # 数値変数に対してPCAを用いて特徴量選択
        data_pca = self.pca.fit_transform(numerical_data)
        # 特徴量選択後のデータを結合 出力：DataFrame
        data_fs = np.concatenate([data_fi, data_pca], axis=1)

        return data_fs, important_features 

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
            data = np.column_stack((data, clusters))  # 'cluster' columnを追加
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
        # 特徴量選択 入力：ndarray 出力: ndarray
        X_fs, self.important_features = self.feature_selection(X_ohe, X_num, y)
        # サブシステムに分割 入力：ndarray　出力：ndarray
        self.attack_data, self.normal_data = self.splitsubsystem(X_fs, y)
        # サンプリング 入力：ndarray　出力：ndarray
        self.sampled_attack = self.make_cluster(self.attack_data)
        self.sampled_normal = self.make_cluster(self.normal_data)
    
        ## training 入力：ndarray
        self.iforest_attack.fit(self.sampled_attack)
        self.iforest_normal.fit(self.sampled_normal)

        
    def predict(self, X):
        predictions = []
        total_points = len(X)

        ## preprocessing
        # one-hotエンコード　入力：DataFrame　出力：ndarray
        X_ohe = self.ohe.transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns, inplace=False).values
        # 正規化　入力：ndarray 出力：ndarray
        X_num = self.mm.transform(X_num) # numeicalデータにだけでよい
        # 特徴量選択 入力：ndarray 出力：ndarray    
        X_ohe = X_ohe[:, self.important_features]
        X_num = self.pca.transform(X_num)
        self.X_processed = np.concatenate([X_ohe, X_num], axis=1)
    
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
