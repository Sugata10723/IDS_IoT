import time
import plotter
import numpy as np
import matplotlib.pyplot as plt
from model_noFS import AnomalyDetector_noFS
from model_hybrid import AnomalyDetector_hybrid
from model_var import AnomalyDetector_var
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class Experiment:
    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train # Pandas DataFrame
        self.X_test = X_test # Pandas DataFrame
        self.y_train = y_train # ndarray
        self.y_test = y_test # ndarray
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

    def run_noFS(self, k):
        model_params = {
            'k': k,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_noFS(**model_params)
        self.fit()
        self.evaluate()
        self.print_results()
        plotter.plot_cluster(self.model.attack_data, self.model.sampled_attack, self.model.normal_data, self.model.sampled_normal)
        plotter.plot_results(self.X_test, self.y_test, self.prediction, self.config)
        plotter.plot_confusion_matrix(self.y_test, self.prediction)

    def run_hybrid(self, k, n_fi, n_pca):
        model_params = {
            'k': k,
            'n_fi': n_fi,
            'n_pca': n_pca,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_hybrid(**model_params)
        self.fit()
        self.evaluate()
        self.print_results()
        plotter.plot_cluster(self.model.attack_data, self.model.sampled_attack, self.model.normal_data, self.model.sampled_normal)
        plotter.plot_results(self.X_test, self.y_test, self.prediction, self.config)
        plotter.plot_confusion_matrix(self.y_test, self.prediction)

    
    def run_var(self, k):
        model_params = {
            'k': k,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_var(**model_params)
        self.fit()
        self.evaluate()
        self.print_results()
        #plotter.plot_cluster(self.model.attack_data, self.model.sampled_attack, self.model.normal_data, self.model.sampled_normal) # 動かない
        plotter.plot_results(self.X_test, self.y_test, self.prediction, self.config)
        plotter.plot_confusion_matrix(self.y_test, self.prediction)






