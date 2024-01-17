import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from umap import UMAP

def plot_cluster(attack_data, attack_sampled, normal_data, normal_sampled):
    umap_plot = UMAP(n_components=2)
    data = np.concatenate([attack_data, normal_data], axis=0)
    # クラスタリング結果をUMAPを使用して2次元にプロット
    data_umap = umap_plot.fit_transform(data)
    attack_umap = umap_plot.transform(attack_data)
    attack_sampled_umap = umap_plot.transform(attack_sampled)
    normal_umap = umap_plot.transform(normal_data)
    normal_sampled_umap = umap_plot.transform(normal_sampled)
    plt.figure(figsize=(10, 7))
    plt.scatter(attack_umap[:, 0], attack_umap[:, 1], c='red', edgecolor='none', s=10, label='Attack')
    plt.scatter(normal_umap[:, 0], normal_umap[:, 1], c='blue', edgecolor='none', s=10, label='Normal')
    plt.scatter(attack_sampled_umap[:, 0], attack_sampled_umap[:, 1], c='orange', edgecolor='none', s=10, label='Attack_sampled')
    plt.scatter(normal_sampled_umap[:, 0], normal_sampled_umap[:, 1], c='green', edgecolor='none', s=10, label='Normal_sampled')
    plt.legend()
    plt.title('sampled data')
    plt.show()


def plot_results(X, y, predictions, config):
    X = X.copy()
    le = LabelEncoder()
    scaler = StandardScaler()
    umap = UMAP(n_components=2)
    for col in config['categorical_columns']:
        X[col] = le.fit_transform(X[col])
    X = scaler.fit_transform(X)
    X_transformed = pd.DataFrame(umap.fit_transform(X))

    # figureを作成
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    #　攻撃と正常のデータをプロット
    colors = ['blue' if label == 0 else 'red' for label in y]
    axs[0].scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='none', s=10)
    classes = ['Normal', 'Attack']
    class_colours = ['blue', 'red']
    recs = [mpatches.Rectangle((0, 0), 1, 1, fc=color) for color in class_colours]
    axs[0].legend(recs, classes, loc=4)
    axs[0].set_title('Data')
    #　予測結果をプロット
    colors = ['blue' if label == 0 else 'red' if label == 1 else 'green' for label in predictions]
    axs[1].scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1], c=colors, edgecolor='none', s=10)
    classes = ['Normal', 'Attack', 'Unknown']
    class_colours = ['blue', 'red', 'green']
    recs = [mpatches.Rectangle((0, 0), 1, 1, fc=color) for color in class_colours]
    axs[1].legend(recs, classes, loc=4)
    axs[1].set_title('Results')
    # figureを表示
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, attack_prd, normal_prd):
    cm = confusion_matrix(y_test, y_pred)
    cm_attack = confusion_matrix(y_test, attack_prd) 
    cm_normal = confusion_matrix(y_test, normal_prd)
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))

    plt.figure(figsize=(36, 10))

    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.subplot(1, 3, 2)
    sns.heatmap(cm_attack, annot=True, fmt='g')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Attack')

    plt.subplot(1, 3, 3)
    sns.heatmap(cm_normal, annot=True, fmt='g')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Normal')

    plt.tight_layout()
    plt.show()

