from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class IoTAnomalyDetector:
    def __init__(self, contamination=0.01, k=20):
        self.contamination = contamination
        self.k = k
        self.cluster_anomaly = None
        self.cluster_normal = None
        self.pca_anomaly = None
        self.pca_normal = None

    def splitsubsystem(self, X, y):
        combined_data = pd.DataFrame(X)
        combined_data['label'] = y
        anomaly_data = combined_data[combined_data['label'] == 1]
        normal_data = combined_data[combined_data['label'] == 0]
        return anomaly_data.drop('label', axis=1), normal_data.drop('label', axis=1)

    def featureselection(self, data):
        #PCAを実行して次元を削減する
        pca = PCA(n_components=30) #根拠のない数字
        data_pca = pd.DataFrame(pca.fit_transform(data))
        return data_pca, pca

    def makeclustereddata(self, data, k):
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(data)
        data['cluster'] = clusters
        cluster_list = [data[data['cluster'] == i].drop('cluster', axis=1) for i in range(k)]

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        data_pca = pd.DataFrame(pca.fit_transform(data.drop('cluster', axis=1)))
        data_pca['cluster'] = clusters

        # Plotting the clusters
        plt.figure(figsize=(10, 7))
        for i in range(k):
            plt.scatter(data_pca[data_pca['cluster'] == i].iloc[:, 0], data_pca[data_pca['cluster'] == i].iloc[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.show()
        
        return cluster_list
            
    def subsystem(self, cluster, contamination, point):
        for df in cluster:
            point_df = pd.DataFrame([point], columns=df.columns)
            iforest = IsolationForest()
            iforest.fit(df)  # fitに用いるのはdfかdf+point_dfか
            prediction = iforest.predict(point_df) #正常なら1を返し、異常なら-1を返す   
            if prediction[0] == 1: # point_dfがdfに含まれるなら1を返す
                return 1
        return 0

    def fit(self, X, y):
        anomaly_data, normal_data = self.splitsubsystem(X, y)
        anomaly_data, self.pca_anomaly = self.featureselection(anomaly_data)
        normal_data, self.pca_normal = self.featureselection(normal_data)

        print("anomaly_data")
        self.cluster_anomaly = self.makeclustereddata(anomaly_data, self.k)
        print("normal_data")
        self.cluster_normal = self.makeclustereddata(normal_data, self.k)

    def predict(self, X):
        anomaly_result = []
        normal_result = []
        predictions = []

        X_anomaly = pd.DataFrame(self.pca_anomaly.transform(X))
        for index, x in X_anomaly.iterrows():
            result = self.subsystem(self.cluster_anomaly, self.contamination, x)
            print(f"Anomaly result for point {index}: {result}")
            anomaly_result.append(result)

        X_normal = pd.DataFrame(self.pca_normal.transform(X))
        for index, x in X_normal.iterrows():
            result = self.subsystem(self.cluster_normal, self.contamination, x)
            print(f"Normal result for point {index}: {result}")
            normal_result.append(result)

        for i in range(len(X)):
            if anomaly_result[i] == 0 and normal_result[i] == 1:
                predictions.append(0)
            elif anomaly_result[i] == 0 and normal_result[i] == 0:
                predictions.append(-1)
            else:
                predictions.append(1)

        # Plotting the results
        pca = PCA(n_components=2)
        X_transformed = pd.DataFrame(pca.fit_transform(X))
        plt.figure(figsize=(10, 7))
        colors = ['blue' if label == 0 else 'red' if label == 1 else 'green' for label in predictions]
        scatter = plt.scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='k')

        # Adding legend
        classes = ['Normal', 'Anomaly', 'Unknown']
        class_colours = ['blue','red', 'green']
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
        plt.legend(recs,classes,loc=4)

        plt.show()

        return predictions
        
        return predictions

