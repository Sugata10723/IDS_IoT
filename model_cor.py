import pandas as pd
import numpy as np
import time
import plotter 
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as preprocessing
from sklearn_extra.cluster import KMedoids


#################################################################################   
# AnomalyDetector_cor
# lableとの相関係数が高い特徴を選択するfeature selection
# パラメータ
# k: クラスタ数
# n_estimators: Isolation Forestの決定木の数
# max_samples: Isolation Forestのサンプル数
# contamination: Isolation Forestの異常スコアの閾値
#################################################################################


class AnomalyDetector_cor:
    def __init__(self, parameters, categorical_columns):
        self.k = parameters['k']
        self.c_attack = parameters['c_attack']
        self.c_normal = parameters['c_normal']
        self.n_estimators = parameters['n_estimators']
        self.max_features = parameters['max_features']
        self.max_samples = parameters['max_samples']
        self.threshold = parameters['threshold']
        self.categorical_columns = categorical_columns
        self.ohe = preprocessing.OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
        self.mm = preprocessing.MinMaxScaler()
        self.iforest_attack = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_attack, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, warm_start=False)
        self.iforest_normal = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, max_features=self.max_features, contamination=self.c_normal, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, warm_start=False)

        self.sampled_attack = None
        self.sampled_normal = None
        self.selected_features = None
        # プロットのため
        self.attack_data = None
        self.normal_data = None
        self.attack_prd = None
        self.normal_prd = None
        self.X_processed = None

    def splitsubsystem(self, X, y):
        attack_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        return X[attack_indices], X[normal_indices]
    
    def feature_selection(self, X, y):
        # 相関係数が高い特徴量を選択
        correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        correlations = np.abs(correlations)
        selected_features = np.argsort(correlations)[::-1]
        selected_features = selected_features[correlations[selected_features] > self.threshold]
        print(selected_features)
        return X[:, selected_features], selected_features

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
        if len(data) > self.k:
            kmeans = MiniBatchKMeans(n_clusters=self.k, init='k-means++', batch_size=100, tol=0.01, n_init=10) 
            clusters = kmeans.fit_predict(data)
            data = np.column_stack((data, clusters))
            data_sampled = self.get_nearest_points(data, kmeans)
            return data_sampled
        else:
            return data
            
    def fit(self, X, y):
        X_ohe = self.ohe.fit_transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns).values
        X_num = self.mm.fit_transform(X_num)
        X_processed = np.concatenate([X_num, X_ohe], axis=1)
        X_processed, self.selected_features = self.feature_selection(X_processed, y)
        self.attack_data, self.normal_data = self.splitsubsystem(X_processed, y)
        self.sampled_attack = self.make_cluster(self.attack_data)
        self.sampled_normal = self.make_cluster(self.normal_data)    
        self.iforest_attack.fit(self.sampled_attack)
        self.iforest_normal.fit(self.sampled_normal)
        
    def predict(self, X):
        attack_results = []
        normal_results = []
        predictions = []
        total_points = len(X)

        X_ohe = self.ohe.transform(X[self.categorical_columns])
        X_num = X.drop(columns=self.categorical_columns).values
        X_num = self.mm.transform(X_num)
        X_processed = np.concatenate([X_num, X_ohe], axis=1)
        X_processed = X_processed[:, self.selected_features]

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

    def plot_anomaly_scores(self):
        scores_attack = self.iforest_attack.decision_function(self.X_processed)
        scores_normal = self.iforest_normal.decision_function(self.X_processed)
        plt.hist(scores_attack, bins=50, alpha=0.5, label='attack')
        plt.hist(scores_normal, bins=50, alpha=0.5, label='normal')
        plt.legend()
        plt.show()
