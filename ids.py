import mlflow
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class IoTAnomalyDetector:
    def __init__(self, k, n_features):
        self.k = k
        self.n_features = n_features
        self.sampled_attack = None
        self.sampled_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        self.pca_attack = None
        self.pca_normal = None
        self.umap_plot = None

    def savepng(self):
        # プロットを保存してMLflowにアーティファクトとしてログし、保存後にファイルを削除
        timestamp = int(time.time())
        plot_file_path = f"results/plot_{timestamp}.png"
        plt.savefig(plot_file_path)
        mlflow.log_artifact(plot_file_path)
        os.remove(plot_file_path)

    def plot_data(self, X, attack_data, normal_data):
        # UMAPを使用してデータを2次元にプロットし、攻撃データと正常データを異なる色で表示
        self.umap_plot = UMAP(n_components=2)
        X_plot = pd.DataFrame(self.umap_plot.fit_transform(X))
        attack_plot = pd.DataFrame(self.umap_plot.transform(attack_data))
        normal_plot = pd.DataFrame(self.umap_plot.transform(normal_data))
        plt.figure(figsize=(10, 7))
        plt.scatter(attack_plot.iloc[:, 0], attack_plot.iloc[:, 1], label='Attack', c='red')
        plt.scatter(normal_plot.iloc[:, 0], normal_plot.iloc[:, 1], label='Normal', c='blue')
        plt.legend()
        plt.title('Attack and Normal Data')
        plt.show()

    def splitsubsystem(self, X, y):
        # データとラベルを結合して、攻撃データと正常データに分割
        combined_data = pd.DataFrame(X)
        combined_data_copy = combined_data.copy()
        combined_data_copy['label'] = y
        attack_data = combined_data_copy[combined_data_copy['label'] == 1]
        normal_data = combined_data_copy[combined_data_copy['label'] == 0]
        attack_data = attack_data.drop('label', axis=1)
        normal_data = normal_data.drop('label', axis=1)
        # プロット
        self.plot_data(X, attack_data, normal_data)
        return attack_data, normal_data

    def featureselection(self, data):
        # PCAを使用して次元を削減
        pca = PCA(n_components=self.n_features)
        data_pca = pd.DataFrame(pca.fit_transform(data))
        return data_pca, pca

    def plot_cluster(self, data, data_sampled, title):
        # クラスタリング結果をUMAPを使用して2次元にプロット
        data_umap = pd.DataFrame(self.umap_plot.fit_transform(data))
        data_sampled_umap = pd.DataFrame(self.umap_plot.transform(data_sampled))
        plt.figure(figsize=(10, 7))
        plt.scatter(data_umap.iloc[:, 0], data_umap.iloc[:, 1], label='Data', c='blue')
        plt.scatter(data_sampled_umap.iloc[:, 0], data_sampled_umap.iloc[:, 1], label='Centroids', c='green')
        plt.legend()
        plt.title(title)
        plt.show()

    def get_nearest_points(self, data, kmeans):
        nearest_points = []
        for i, center in enumerate(kmeans.cluster_centers_):
            distances = data[data['cluster'] == i].apply(lambda x: distance.euclidean(x[:-1], center), axis=1)
            nearest_point = distances.idxmin()
            nearest_points.append(data.loc[nearest_point])
        nearest_points = pd.DataFrame(nearest_points)
        nearest_points = nearest_points.drop('cluster', axis=1)
        return nearest_points

    def make_cluster(self, data, k):
        # K-meansを使用してクラスタリング
        data = data.copy()
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(data)
        data['cluster'] = clusters
        # 各クラスタの中心点を取得してサンプリング
        data_sampled = self.get_nearest_points(data, kmeans)
        return data_sampled

    def subsystem(self, data_sampled, iforest, point):
        # Isolation Forestでポイントの異常検知を行う
        point_df = pd.DataFrame([point], columns=data_sampled.columns)
        prediction = iforest.predict(point_df)
        if prediction[0] == 1:
            return 1  # 正常なら1を返す
        return 0  # 異常なら0を返す

    def fit(self, X, y):
        # 攻撃データと正常データに分割
        attack_data, normal_data = self.splitsubsystem(X, y)
        # PCAを使用して次元削減
        attack_data, self.pca_attack = self.featureselection(attack_data)
        normal_data, self.pca_normal = self.featureselection(normal_data)
        # K-meansでクラスタリングし、Isolation Forestで学習
        self.sampled_attack = self.make_cluster(attack_data, self.k)
        self.sampled_normal = self.make_cluster(normal_data, self.k)
        self.iforest_attack = IsolationForest(n_estimators=50, max_samples=100).fit(self.sampled_attack)
        self.iforest_normal = IsolationForest(n_estimators=50, max_samples=100).fit(self.sampled_normal)
        
        self.plot_cluster(attack_data, self.sampled_attack, 'Attack Data')
        self.plot_cluster(normal_data, self.sampled_normal, 'Normal Data')


    def plot_results(self, X, predictions):
        # 学習結果をUMAPを使用して2次元にプロット
        X_transformed = pd.DataFrame(self.umap_plot.fit_transform(X))
        plt.figure(figsize=(10, 7))
        colors = ['blue' if label == 0 else 'red' if label == 1 else 'green' for label in predictions]
        scatter = plt.scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='none')
        classes = ['Normal', 'Attack', 'Unknown']
        class_colours = ['blue', 'red', 'green']
        recs = []
        for i in range(0, len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
        plt.legend(recs, classes, loc=4)
        plt.title('Results')
        plt.show()

    def predict(self, X):
        attack_results = []
        normal_results = []
        predictions = []
        total_points = len(X)
        # PCAを使用して次元を削減
        X_attack = pd.DataFrame(self.pca_attack.transform(X))
        X_normal = pd.DataFrame(self.pca_normal.transform(X))
        # 攻撃データのサブシステムで判定
        for index, x in X_attack.iterrows():
            result = self.subsystem(self.sampled_attack, self.iforest_attack, x)
            attack_results.append(result)
            anomaly_score_attack = self.iforest_attack.decision_function(x.values.reshape(1, -1))
            print(f"\rProgress: {(((index+1) / total_points)/ 2) * 100:.2f}%", end="")
        # 正常データのサブシステムで判定
        for index, x in X_normal.iterrows():
            result = self.subsystem(self.sampled_normal, self.iforest_normal, x)
            normal_results.append(result)
            anomaly_score_normal = self.iforest_normal.decision_function(x.values.reshape(1, -1))
            print(f"\rProgress: {(((index+1) / total_points)/ 2) * 100 + 50:.2f}%", end="")
        # predictionsを作成
        for i in range(total_points):
            if attack_results[i] == 0 and normal_results[i] == 1:
                predictions.append(0)  # normal
            elif attack_results[i] == 0 and normal_results[i] == 0:
                predictions.append(-1)  # unknown
            else:
                predictions.append(1)  # attack
        # 結果をプロット
        self.plot_results(X, predictions)
        print(predictions)
        print(attack_results)
        print(normal_results)

        return predictions
