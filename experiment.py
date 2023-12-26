import time
import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from model import AnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from umap import UMAP

class Experiment:
    def __init__(self, data, labels, config):
        self.data = data
        self.labels = labels
        self.config = config
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        fit_time = time.perf_counter() - start_time
        return fit_time 

    def evaluate(self):
        start_time = time.perf_counter()
        y_pred = self.model.predict(self.X_test)
        evaluate_time = time.perf_counter() - start_time
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted') # 2値分類問題だが、y_predに-1が含まれるためweightedを使用
        self.predictions = y_pred

        return evaluate_time, accuracy, f1

    def plot_data(self, attack_data, normal_data):
        umap_plot = UMAP(n_components=2)
        # UMAPを使用してデータを2次元にプロットし、攻撃データと正常データを異なる色で表示
        X_plot = pd.DataFrame(umap_plot.fit_transform(self.data))
        attack_plot = pd.DataFrame(umap_plot.transform(attack_data))
        normal_plot = pd.DataFrame(umap_plot.transform(normal_data))
        plt.figure(figsize=(10, 7))
        plt.scatter(attack_plot.iloc[:, 0], attack_plot.iloc[:, 1], label='Attack', c='red', s=10)
        plt.scatter(normal_plot.iloc[:, 0], normal_plot.iloc[:, 1], label='Normal', c='blue', s=10)
        plt.legend()
        plt.title('Attack and Normal Data')
        plt.show()

    def plot_cluster(self, data, data_sampled, title):
        umap_plot = UMAP(n_components=2)
        # クラスタリング結果をUMAPを使用して2次元にプロット
        data_umap = pd.DataFrame(umap_plot.fit_transform(data))
        data_sampled_umap = pd.DataFrame(umap_plot.transform(data_sampled))
        plt.figure(figsize=(10, 7))
        plt.scatter(data_umap.iloc[:, 0], data_umap.iloc[:, 1], label='Data', c='blue', s=10)
        plt.scatter(data_sampled_umap.iloc[:, 0], data_sampled_umap.iloc[:, 1], label='Centroids', c='green', s=10)
        plt.legend()
        plt.title(title)
        plt.show()
    
    def plot_anomaly_scores(self, anomaly_scores_a, anomaly_scores_n):
        # 異常スコアをプロット
        plt.figure(figsize=(10, 7))
        plt.hist(anomaly_scores_a, bins=100, label='Attack', color='red', alpha=0.5)
        plt.hist(anomaly_scores_n, bins=100, label='Normal', color='blue', alpha=0.5)
        plt.legend()
        plt.title('Anomaly Scores')
        plt.show()

    def plot_results(self, X, predictions):
        umap_plot = UMAP(n_components=2)
        # 学習結果をUMAPを使用して2次元にプロット
        X_transformed = pd.DataFrame(umap_plot.fit_transform(X))
        plt.figure(figsize=(10, 7))
        colors = ['blue' if label == 0 else 'red' if label == 1 else 'green' for label in predictions]
        scatter = plt.scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='none', s=10)
        classes = ['Normal', 'Attack', 'Unknown']
        class_colours = ['blue', 'red', 'green']
        recs = []
        for i in range(0, len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
        plt.legend(recs, classes, loc=4)
        plt.title('Results')
        plt.show()

    def plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        unique_labels = np.unique(np.concatenate([self.y_test, y_pred]))
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def print_results(self, fit_time, evaluate_time, accuracy, f1):
        # 結果を出力
        print('Fit time: {:.4f}'.format(fit_time))
        print('Evaluate time: {:.4f}'.format(evaluate_time))
        print('Accuracy: {:.4f}'.format(accuracy))
        print('F1: {:.4f}'.format(f1))

    def plot_graphs(self):
        attack_data, normal_data = self.model.splitsubsystem(self.X_train, self.y_train)
        self.plot_data(attack_data, normal_data)
        self.plot_cluster(self.model.attack_data, self.model.sampled_attack, 'Attack Data')
        self.plot_cluster(self.model.normal_data, self.model.sampled_normal, 'Normal Data')
        self.plot_results(self.X_test, self.predictions)
        self.plot_anomaly_scores(self.model.anomaly_scores_a, self.model.anomaly_scores_n)
        self.plot_confusion_matrix(self.predictions)

    def run(self, model_params=None):
        self.model = AnomalyDetector(**model_params)
        self.data = Preprocessor(self.data, self.config).process()
        self.split_data()
        fit_time = self.fit()
        evaluate_time, accuracy, f1 = self.evaluate()

        self.print_results(fit_time, evaluate_time, accuracy, f1)
        #self.plot_graphs()




