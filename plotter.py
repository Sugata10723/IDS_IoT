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
import pandas as pd
from umap import UMAP
import seaborn as sns
import sweetviz as sv

def plot_cluster(attack_data, attack_sampled, normal_data, normal_sampled):
    umap_plot = UMAP(n_components=2)
    data = np.concatenate([attack_data, normal_data], axis=0)
    # クラスタリング結果をUMAPを使用して2次元にプロット
    data_umap = umap_plot.fit_transform(data)
    attack_sampled_umap = umap_plot.transform(attack_sampled)
    normal_sampled_umap = umap_plot.transform(normal_sampled)
    plt.figure(figsize=(10, 7))
    plt.scatter(data_umap[:, 0], data_umap[:, 1], c='blue', edgecolor='none', s=10, label='data')
    plt.scatter(attack_sampled_umap[:, 0], attack_sampled_umap[:, 1], c='red', edgecolor='none', s=10, label='attack_sampled')
    plt.scatter(normal_sampled_umap[:, 0], normal_sampled_umap[:, 1], c='green', edgecolor='none', s=10, label='normal_sampled')
    plt.legend()
    plt.title('sampled data')
    plt.show()

def plot_anomaly_scores(anomaly_scores_a, anomaly_scores_n):
    # 異常スコアをプロット
    plt.figure(figsize=(10, 7))
    plt.hist(anomaly_scores_a, bins=100, label='Attack', color='red', alpha=0.5)
    plt.hist(anomaly_scores_n, bins=100, label='Normal', color='blue', alpha=0.5)
    plt.legend()
    plt.title('Anomaly Scores')
    plt.show()

def plot_results(X, y, predictions, config):
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

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_feature_importances(X, y, categorical_columns, top_n=10):
    import xgboost as xgb
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import preprocessing

    # カテゴリカル変数をOneHotEncoding
    ohe = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    X_ohe = ohe.fit_transform(X[categorical_columns])
    feature_names = ohe.get_feature_names_out(categorical_columns)
    feature_names = np.concatenate([X.drop(columns=categorical_columns).columns, feature_names])
    X = np.concatenate([X.drop(columns=categorical_columns).values, X_ohe], axis=1)

    # 正規化(正規化の有無で結果は変わらない)
    mm = preprocessing.MinMaxScaler()
    X = mm.fit_transform(X)

    # FIを用いて特徴量選択
    xgb_model = xgb.XGBRegressor() # xgboostを使用
    xgb_model.fit(X, y)
    feature_importances = xgb_model.feature_importances_

    # 特徴量の重要度が高い上位N個を取得
    indices = np.argsort(feature_importances)[-top_n:]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_feature_importances = feature_importances[indices]

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names)
    plt.title('Feature Importances')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.show()

def plot_heatmap(scores, title):
    # F1スコアのヒートマップをプロット
    plt.figure(figsize=(10, 10))
    plt.imshow(scores, cmap='hot', interpolation='nearest', origin='lower', vmax=1)
    plt.colorbar(label=title)
    plt.xlabel('n_pcas')
    plt.ylabel('n_fis')
    plt.title(f'{title} Heatmap')
    plt.show()

