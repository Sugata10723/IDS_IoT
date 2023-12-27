import time
import pandas as pd
import numpy as np
from transformer import Transformer
from model import AnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree
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
        self.accuracy = None
        self.f1 = None
        # パラメータを変えて繰り返し実験するために、preprocessとsplitをコンストラクタで実行
        self.data = Transformer(self.data, self.config).transform()
        self.split_data()

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
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred, average='weighted') # 2値分類問題だが、y_predに-1が含まれるためweightedを使用
        self.predictions = y_pred

        return evaluate_time

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

    def plot_estimators(self):
        estimator_attack = self.model.iforest_attack.estimators_[0] # 1つ目の決定木を取得
        estimator_normal = self.model.iforest_normal.estimators_[0]

        plt.figure(figsize=(20, 10))
        plot_tree(estimator_attack, filled=True)
        plt.title('Attack Estimator')
        plt.show()
        plt.figure(figsize=(20, 10))
        plot_tree(estimator_normal, filled=True)
        plt.title('Normal Estimator')
        plt.show()

    def plot_feature_importances(self):
        n_features = self.model.n_features
        feature_importances_attack = np.zeros(n_features)
        for tree in self.model.iforest_attack.estimators_:
            for node in tree.tree_.__getstate__()['nodes']:
                if node[0] != -1:  # 葉ノードではない場合
                    feature_importances_attack[node[2]] += 1  # node[2]は分割に使用された特徴量のインデックス
        feature_importances_attack /= len(self.model.iforest_attack.estimators_) 
        plt.figure(figsize=(10, 7))
        plt.bar(range(n_features), feature_importances_attack)
        plt.title('Attack Feature Importances')
        plt.show()

        feature_importances_normal = np.zeros(n_features)
        for tree in self.model.iforest_normal.estimators_:
            for node in tree.tree_.__getstate__()['nodes']:
                if node[0] != -1:
                    feature_importances_normal[node[2]] += 1
        feature_importances_normal /= len(self.model.iforest_normal.estimators_)
        plt.figure(figsize=(10, 7))
        plt.bar(range(n_features), feature_importances_normal)
        plt.title('Normal Feature Importances')
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

    def plot_feature_distribution(self, data, n_features, title):
        n_rows = 5
        n_cols = 6
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            axs[row, col].hist(data.iloc[:,i])
            axs[row, col].set_title("Feature " + str(i))
            axs[row, col].set_xlabel("Value")
            axs[row, col].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def print_results(self, fit_time, evaluate_time):
        # 結果を出力
        print('Fit time: {:.4f}'.format(fit_time))
        print('Evaluate time: {:.4f}'.format(evaluate_time))
        print('Accuracy: {:.4f}'.format(self.accuracy))
        print('F1: {:.4f}'.format(self.f1))

    def plot_graphs(self):
        attack_data, normal_data = self.model.splitsubsystem(self.X_train, self.y_train)
        #self.plot_data(attack_data, normal_data)
        #self.plot_cluster(self.model.attack_data, self.model.sampled_attack, 'Attack Data')
        #self.plot_cluster(self.model.normal_data, self.model.sampled_normal, 'Normal Data')
        #self.plot_results(self.X_test, self.predictions)
        self.plot_anomaly_scores(self.model.anomaly_scores_a, self.model.anomaly_scores_n)
        self.plot_confusion_matrix(self.predictions)
        #self.plot_estimators()
        #self.plot_feature_importances()

    def run(self, model_params=None):
        self.model = AnomalyDetector(**model_params)
        fit_time = self.fit()
        evaluate_time = self.evaluate()

        self.print_results(fit_time, evaluate_time)
        self.plot_graphs()




