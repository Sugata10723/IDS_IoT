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
    from umap import UMAP
    import matplotlib.pyplot as plt
    import numpy as np

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


def plot_estimators():
    estimator_attack = model.iforest_attack.estimators_[0] # 1つ目の決定木を取得
    estimator_normal = model.iforest_normal.estimators_[0]

    plt.figure(figsize=(20, 10))
    plot_tree(estimator_attack, filled=True)
    plt.title('Attack Estimator')
    plt.show()
    plt.figure(figsize=(20, 10))
    plot_tree(estimator_normal, filled=True)
    plt.title('Normal Estimator')
    plt.show()

def plot_feature_importances():
    n_features = model.n_features
    feature_importances_attack = np.zeros(n_features)
    for tree in model.iforest_attack.estimators_:
        for node in tree.tree_.__getstate__()['nodes']:
            if node[0] != -1:  # 葉ノードではない場合
                feature_importances_attack[node[2]] += 1  # node[2]は分割に使用された特徴量のインデックス
    feature_importances_attack /= len(model.iforest_attack.estimators_) 
    plt.figure(figsize=(10, 7))
    plt.bar(range(n_features), feature_importances_attack)
    plt.title('Attack Feature Importances')
    plt.show()

    feature_importances_normal = np.zeros(n_features)
    for tree in model.iforest_normal.estimators_:
        for node in tree.tree_.__getstate__()['nodes']:
            if node[0] != -1:
                feature_importances_normal[node[2]] += 1
    feature_importances_normal /= len(model.iforest_normal.estimators_)
    plt.figure(figsize=(10, 7))
    plt.bar(range(n_features), feature_importances_normal)
    plt.title('Normal Feature Importances')
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

def plot_feature_distribution(data, n_features, title):
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

def plot_pca_components():
    n_components = model.pca_attack.components_.shape[0]
    n_features = model.pca_attack.components_.shape[1]

    fig, axs = plt.subplots(n_components, 2, figsize=(10, 6 * n_components))

    for i in range(n_components):
        axs[i, 0].bar(np.arange(n_features), model.pca_attack.components_[i])
        axs[i, 0].set_title(f'PCA attack component {i+1}')
        axs[i, 0].set_xlabel('Feature')
        axs[i, 0].set_ylabel('Value')

        axs[i, 1].bar(np.arange(n_features), model.pca_normal.components_[i])
        axs[i, 1].set_title(f'PCA normal component {i+1}')
        axs[i, 1].set_xlabel('Feature')
        axs[i, 1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

def plot_pca_components_ave():
    n_features = model.pca_attack.components_.shape[1]

    avg_attack_components = model.pca_attack.components_.mean(axis=0)
    avg_normal_components = model.pca_normal.components_.mean(axis=0)

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

def plot_feature_importances(X, y, categorical_columns, top_n=10):
    import xgboost as xgb
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OneHotEncoder

    # OneHotEncoderのインスタンスを作成
    ohe = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')

    # one-hotエンコード 入力：DataFrame　出力：ndarray
    X_ohe = ohe.fit_transform(X[categorical_columns])

    # エンコード後の特徴量名を取得
    feature_names = ohe.get_feature_names_out(categorical_columns)

    # エンコードされていない特徴量名と結合
    feature_names = np.concatenate([X.drop(columns=categorical_columns).columns, feature_names])

    # エンコードされたデータと元のデータを結合
    X = np.concatenate([X.drop(columns=categorical_columns).values, X_ohe], axis=1)

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

def plot_sweetviz(X_train, X_test):
    report = sv.compare([X_train, "Train Data"], [X_test, "Test Data"])
    report.show_notebook()


