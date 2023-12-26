import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class AnomalyDetector:
    def __init__(self, k, n_features):
        self.k = k
        self.n_features = n_features
        self.sampled_attack = None
        self.sampled_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        self.pca_attack = None
        self.pca_normal = None

        # for plotting
        self.attack_data = None
        self.normal_data = None
        self.anomaly_scores_a = None
        self.anomaly_scores_n = None

    def splitsubsystem(self, X, y):
        # 攻撃データと正常データに分割
        combined_data = pd.DataFrame(X)
        combined_data_copy = combined_data.copy()
        combined_data_copy['label'] = y
        attack_data = combined_data_copy[combined_data_copy['label'] == 1]
        normal_data = combined_data_copy[combined_data_copy['label'] == 0]
        attack_data = attack_data.drop('label', axis=1)
        normal_data = normal_data.drop('label', axis=1)

        return attack_data, normal_data

    def featureselection(self, data):
        # PCAを使用して次元を削減
        pca = PCA(n_components=self.n_features)
        data_pca = pd.DataFrame(pca.fit_transform(data))
        return data_pca, pca

    def get_nearest_points(self, data, kmeans):
        # 各クラスタの中心点に最も近い点を取得
        nearest_points = []
        for i, center in enumerate(kmeans.cluster_centers_):
            distances = data[data['cluster'] == i].apply(lambda x: distance.euclidean(x[:-1], center), axis=1)
            nearest_point = distances.idxmin()
            nearest_points.append(data.loc[nearest_point])
        nearest_points = pd.DataFrame(nearest_points)
        nearest_points = nearest_points.drop('cluster', axis=1)
        return nearest_points

    def make_cluster(self, data, k):
        # K-meansを使用してデータをサンプリング
        data = data.copy()
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(data)
        data['cluster'] = clusters
        data_sampled = self.get_nearest_points(data, kmeans)
        return data_sampled

    def fit(self, X, y):
        attack_data, normal_data = self.splitsubsystem(X, y)

        self.attack_data, self.pca_attack = self.featureselection(attack_data)
        self.normal_data, self.pca_normal = self.featureselection(normal_data)
        
        self.sampled_attack = self.make_cluster(self.attack_data, self.k)
        self.sampled_normal = self.make_cluster(self.normal_data, self.k)
        
        self.iforest_attack = IsolationForest(n_estimators=50, max_samples=100).fit(self.sampled_attack)
        self.iforest_normal = IsolationForest(n_estimators=50, max_samples=100).fit(self.sampled_normal)
        

    def predict(self, X):
        attack_results = []
        normal_results = []
        predictions = []
        total_points = len(X)
        X_attack = pd.DataFrame(self.pca_attack.transform(X))
        X_normal = pd.DataFrame(self.pca_normal.transform(X))

        attack_results = self.iforest_attack.predict(X_attack)
        attack_results = [1 if result == 1 else 0 for result in attack_results]   
        self.anomaly_scores_a = self.iforest_attack.decision_function(X_attack) 
        
        normal_results = self.iforest_normal.predict(X_normal)
        normal_results = [1 if result == 1 else 0 for result in normal_results]
        self.anomaly_scores_n = self.iforest_normal.decision_function(X_normal)
        
        for i in range(total_points):
            if attack_results[i] == 0 and normal_results[i] == 1:
                predictions.append(0)  # normal
            elif attack_results[i] == 0 and normal_results[i] == 0:
                predictions.append(-1)  # unknown
            else:
                predictions.append(1)  # attack

        return predictions
