import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from umap import UMAP


class Plot:
    def __init__(self, data, model, y_test):
        self.data = data
        self.model = model
        self.y_test = y_test

        self.attack_data = None # pcaを行った後のデータ
        self.normal_data = None
        self.anomaly_scores_a = None 
        self.anomaly_scores_n = None
        self.X_attack = None

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

    def plot_pca_components(self):
        n_components = self.model.pca_attack.components_.shape[0]
        n_features = self.model.pca_attack.components_.shape[1]

        fig, axs = plt.subplots(n_components, 2, figsize=(10, 6 * n_components))

        for i in range(n_components):
            axs[i, 0].bar(np.arange(n_features), self.model.pca_attack.components_[i])
            axs[i, 0].set_title(f'PCA attack component {i+1}')
            axs[i, 0].set_xlabel('Feature')
            axs[i, 0].set_ylabel('Value')

            axs[i, 1].bar(np.arange(n_features), self.model.pca_normal.components_[i])
            axs[i, 1].set_title(f'PCA normal component {i+1}')
            axs[i, 1].set_xlabel('Feature')
            axs[i, 1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def plot_pca_components_ave(self):
        n_features = self.model.pca_attack.components_.shape[1]

        avg_attack_components = self.model.pca_attack.components_.mean(axis=0)
        avg_normal_components = self.model.pca_normal.components_.mean(axis=0)

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.bar(np.arange(n_features), avg_attack_components)
        plt.title('Average PCA attack components')
        plt.xlabel('Feature')
        plt.ylabel('Value')

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(n_features), avg_normal_components)
        plt.title('Average PCA normal components')
        plt.xlabel('Feature')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.show()

    def plot_graphs(self):
        attack_data, normal_data = self.model.splitsubsystem(self.X_train, self.y_train)
        self.plot_data(attack_data, normal_data)
        self.plot_cluster(self.model.attack_data, self.model.sampled_attack, 'Attack Data')
        self.plot_cluster(self.model.normal_data, self.model.sampled_normal, 'Normal Data')
        self.plot_results(self.X_test, self.predictions)
        self.plot_anomaly_scores(self.model.anomaly_scores_a, self.model.anomaly_scores_n)
        self.plot_confusion_matrix(self.predictions)
        self.plot_estimators()
        self.plot_feature_importances()
