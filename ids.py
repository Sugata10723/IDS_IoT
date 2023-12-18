import os
import mlflow
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class IoTAnomalyDetector:
    def __init__(self, k, n_features):
        self.k = k
        self.n_features = n_features
        self.cluster_attack = None
        self.cluster_normal = None
        self.iforest_attack = None
        self.iforest_normal = None
        self.pca_attack = None
        self.pca_normal = None
        self.umap_plot = None

    def savepng(self):
        # Save the plot as an image file
        timestamp = int(time.time())
        plot_file_path = f"results/plot_{timestamp}.png"
        plt.savefig(plot_file_path)

        # Log the image file to MLflow
        mlflow.log_artifact(plot_file_path)

        # Remove the image file if not needed anymore
        os.remove(plot_file_path)

    def plot_data(self, X, attack_data, normal_data):
        self.umap_plot = UMAP(n_components=2)
        X_plot = pd.DataFrame(self.umap_plot.fit_transform(X))
        attack_plot = pd.DataFrame(self.umap_plot.transform(attack_data))
        normal_plot = pd.DataFrame(self.umap_plot.transform(normal_data))

        plt.figure(figsize=(10, 7))
        plt.scatter(attack_plot.iloc[:, 0], attack_plot.iloc[:, 1], label='Attack')
        plt.scatter(normal_plot.iloc[:, 0], normal_plot.iloc[:, 1], label='Normal')
        plt.legend()
        plt.title('Attack and Normal Data')
        self.savepng()
        plt.show()

    def splitsubsystem(self, X, y):
        combined_data = pd.DataFrame(X)
        combined_data_copy = combined_data.copy()  # Copy the DataFrame
        combined_data_copy['label'] = y
        
        attack_data = combined_data_copy[combined_data_copy['label'] == 1]
        normal_data = combined_data_copy[combined_data_copy['label'] == 0]
        
        attack_data = attack_data.drop('label', axis=1)
        normal_data = normal_data.drop('label', axis=1)

        # Plotting the data
        self.plot_data(X, attack_data, normal_data)

        return attack_data, normal_data

    def featureselection(self, data):
        # PCAを実行して次元を削減する
        pca = PCA(n_components=self.n_features) 
        data_PCA = pd.DataFrame(pca.fit_transform(data))
        return data_PCA, pca

    def plot_cluster(self, data, clusters, k, title):
        # Plotting the data
        data_umap = pd.DataFrame(self.umap_plot.fit_transform(data.drop('cluster', axis=1)))
        data_umap['cluster'] = clusters

        # Plotting the clusters
        plt.figure(figsize=(10, 7))
        for i in range(k):
            plt.scatter(data_umap[data_umap['cluster'] == i].iloc[:, 0], data_umap[data_umap['cluster'] == i].iloc[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.title(title)
        self.savepng()
        plt.show()

    def makeclustereddata(self, data, k, title):
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(data)
        data['cluster'] = clusters
        cluster_list = [data[data['cluster'] == i].drop('cluster', axis=1) for i in range(k)]

        # Plotting the clusters
        self.plot_cluster(data, clusters, k, title)
        
        return cluster_list

    def subsystem(self, cluster, iforest_list, point):
        for df, iforest in zip(cluster, iforest_list):
            point_df = pd.DataFrame([point], columns=df.columns)
            prediction = iforest.predict(point_df)  # 正常なら1を返し、異常なら-1を返す   
            if prediction[0] == 1:  # point_dfがdfに含まれるなら1を返す
                return 1
        return 0

    def fit(self, X, y):
        attack_data, normal_data = self.splitsubsystem(X, y)
        attack_data, self.pca_attack = self.featureselection(attack_data)
        normal_data, self.pca_normal = self.featureselection(normal_data)

        self.cluster_attack = self.makeclustereddata(attack_data, self.k, "Attack_data")
        self.cluster_normal = self.makeclustereddata(normal_data, self.k, "Normal_data")

        self.iforest_attack = [IsolationForest().fit(df) for df in self.cluster_attack]
        self.iforest_normal = [IsolationForest().fit(df) for df in self.cluster_normal]

    def plot_results(self, X, predictions):
        # Plotting the results
        X_transformed = pd.DataFrame(self.umap_plot.fit_transform(X))
        plt.figure(figsize=(10, 7))
        colors = ['blue' if label == 0 else 'red' if label == 1 else 'green' for label in predictions]
        scatter = plt.scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='k')

        # Adding legend
        classes = ['Normal', 'Attack', 'Unknown']
        class_colours = ['blue', 'red', 'green']
        recs = []
        for i in range(0, len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
        plt.legend(recs, classes, loc=4)
        plt.title('Results')
        self.savepng()
        plt.show()

    def predict(self, X):
        attack_result = []
        normal_result = []
        predictions = []

        # attackのサブシステムで判定
        X_attack = pd.DataFrame(self.pca_attack.transform(X))
        total_points = len(X_attack)
        for index, x in X_attack.iterrows():
            result = self.subsystem(self.cluster_attack, self.iforest_attack, x)
            attack_result.append(result)
            # 学習の進行状況を表示
            print(f"\rProgress: {(((index+1) / total_points)/ 2) * 100:.2f}%", end="")

        # normalのサブシステムで判定
        X_normal = pd.DataFrame(self.pca_normal.transform(X))
        for index, x in X_normal.iterrows():
            result = self.subsystem(self.cluster_normal, self.iforest_normal, x)
            normal_result.append(result)
            # 学習の進行状況を表示
            print(f"\rProgress: {(((index+1) / total_points)/ 2) * 100 + 50:.2f}%", end="")

        # predictionsを作成
        for i in range(len(X)):
            if attack_result[i] == 0 and normal_result[i] == 1:
                predictions.append(0)
            elif attack_result[i] == 0 and normal_result[i] == 0:
                predictions.append(-1)
            else:
                predictions.append(1)

        # Plotting the results
        self.plot_results(X, predictions)
        # 学習が終わったら音を出す
        os.system('say "学習が終わりました"')  # Macの場合
        print(predictions)

        return predictions
