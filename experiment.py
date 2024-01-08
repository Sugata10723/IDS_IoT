import time
import plotter
import numpy as np
import matplotlib.pyplot as plt
from model import AnomalyDetector
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Experiment:
    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train # Pandas DataFrame
        self.X_test = X_test # Pandas DataFrame
        self.y_train = y_train # Pandas Series
        self.y_test = y_test # Pandas Series
        self.config = config
        self.model = None
        self.accuracy = None
        self.f1 = None
        self.fit_time = None
        self.evaluate_time = None
        # プロットのため
        self.prediction = None

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train) # X_train: Pandas DataFrame, y_train: NumPy Array 
        self.fit_time = time.perf_counter() - start_time 

    def evaluate(self):
        start_time = time.perf_counter()
        self.prediction = self.model.predict(self.X_test) # X_test: Panda DataFrame
        self.evaluate_time = time.perf_counter() - start_time
        
        self.accuracy = accuracy_score(self.y_test, self.prediction)
        self.f1 = f1_score(self.y_test, self.prediction, average='weighted') # 2値分類問題だが、y_predに-1が含まれるためweightedを使用

    def print_results(self):
        print('Fit time: {:.4f}'.format(self.fit_time))
        print('Evaluate time: {:.4f}'.format(self.evaluate_time))
        print('Accuracy: {:.4f}'.format(self.accuracy))
        print('F1: {:.4f}'.format(self.f1))

    def run(self, k, n_fi, n_pca):
        model_params = {
            'k': k,
            'n_fi': n_fi,
            'n_pca': n_pca,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector(**model_params)
        self.fit()
        self.evaluate()
        self.print_results()
        #plotter.plot_results(self.X_test, self.y_test, self.prediction, self.config)
        #plotter.plot_confusion_matrix(self.y_test, self.prediction)

    def grid_run(self, k, max_feature, dif):
        n_fis = list(range(1, max_feature + 1, dif))
        n_pcas = list(range(1, max_feature + 1, dif))
        param_grid = [{'n_fi': n_fi, 'n_pca': n_pca} for n_fi in n_fis for n_pca in n_pcas if n_fi + n_pca <= max_feature]
        f1_scores = np.zeros((max_feature, max_feature))
        accuracy_scores = np.zeros((max_feature, max_feature))

        for params in param_grid:
            print(params)
            self.run(k, n_fi=params['n_fi'], n_pca=params['n_pca'])
            print("-----------------------------------------------------")
            f1_scores[params['n_fi'] - 1, params['n_pca'] - 1] = self.f1
            accuracy_scores[params['n_fi'] - 1, params['n_pca'] - 1] = self.accuracy

        # F1スコアのヒートマップをプロット
        plt.figure(figsize=(10, 10))
        plt.imshow(f1_scores, cmap='hot', interpolation='nearest', origin='lower', vmax=1)
        plt.colorbar(label='F1 Score')
        plt.xlabel('n_pcas')
        plt.ylabel('n_fis')
        plt.title('F1 Score Heatmap')
        plt.show()

        # Accuracyのヒートマップをプロット
        plt.figure(figsize=(10, 10))
        plt.imshow(accuracy_scores, cmap='hot', interpolation='nearest', origin='lower', vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xlabel('n_pcas')
        plt.ylabel('n_fis')
        plt.title('Accuracy Heatmap')
        plt.show()

    def k_run(self, min_k, max_k, dif, n_fi, n_pca):
        aucs = []
        f1s = []
        for i in range(min_k, max_k + dif, dif):
            self.run(i, n_fi=n_fi, n_pca=n_pca)
            aucs.append(self.accuracy)
            f1s.append(self.f1)

        plt.figure(figsize=(10, 7))
        plt.plot(range(min_k, max_k + dif, dif), aucs, label='AUC') 
        plt.plot(range(min_k, max_k + dif, dif), f1s, label='F1 Score')

        plt.title('AUC and F1 Score over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.legend()

        plt.show()






