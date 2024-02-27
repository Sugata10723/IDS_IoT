import time
import plotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_noFS import AnomalyDetector_noFS
from model_hybrid import AnomalyDetector_hybrid
from model_var import AnomalyDetector_var
from model_mean import AnomalyDetector_mean
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
        self.accuracy = None
        self.f1 = None
        self.evaluate_time_per_data = None

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train) # X_train: Pandas DataFrame, y_train: NumPy Array 
        self.fit_time = time.perf_counter() - start_time 

    def evaluate(self):
        start_time = time.perf_counter()
        self.prediction = self.model.predict(self.X_test) # X_test: Panda DataFrame
        self.evaluate_time = time.perf_counter() - start_time

    def print_results(self, title):
        self.accuracy, attack_acu, normal_acu = (metric for metric in 
                                            [accuracy_score(self.y_test, pred) for pred in 
                                            [self.prediction, self.model.attack_prd, self.model.normal_prd]])
        self.f1, f1_attack, f1_normal = (metric for metric in 
                                    [f1_score(self.y_test, pred, average='weighted') for pred in 
                                    [self.prediction, self.model.attack_prd, self.model.normal_prd]])
        fit_time, self.evaluate_time_per_data = self.fit_time, self.evaluate_time / self.X_test.shape[0] * 1000000
        print("------------------------------")
        print(f"Feature selection: {title} k:{self.model.k}")
        print(f"Accuracy: {format(self.accuracy, '.3f')}")
        print(f"Accuracy in Attack Subsystem: {format(attack_acu, '.3f')}")
        print(f"Accuracy in Normal Subsystem: {format(normal_acu, '.3f')}")
        print(f"F1 Score: {format(self.f1, '.3f')}")
        print(f"F1 Score in Attack Subsystem: {format(f1_attack, '.3f')}")
        print(f"F1 Score in Normal Subsystem: {format(f1_normal, '.3f')}")
        print(f"Fit Time: {format(fit_time, '.1f')}s")
        print(f"Evaluate Time per Data: {format(self.evaluate_time_per_data, '.1f')}us")
        print("------------------------------")

    def run_noFS(self, k, c_attack, c_normal):
        model_params = {
            'k': k,
            'c_attack': c_attack,
            'c_normal': c_normal,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_noFS(**model_params)
        self.fit()
        self.evaluate()
        self.print_results("noFS")
        #plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
        return self.evaluate_time_per_data, self.accuracy, self.f1

    def run_mean(self, k, n_features, c_attack, c_normal):
        model_params = {
            'k': k,
            'n_features': n_features,
            'c_attack': c_attack,
            'c_normal': c_normal,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_mean(**model_params)
        self.fit()
        self.evaluate()
        self.print_results("mean")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
        self.model.plot_anomaly_scores()
        return self.evaluate_time_per_data, self.accuracy, self.f1

    def run_hybrid(self, k, n_fi, n_pca, c_attack, c_normal):
        model_params = {
            'k': k,
            'n_fi': n_fi,
            'n_pca': n_pca,
            'c_attack': c_attack,
            'c_normal': c_normal,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_hybrid(**model_params)
        self.fit()
        self.evaluate()
        self.print_results(f"hybrid, n_fi={n_fi}, n_pca={n_pca}")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
        self.model.plot_anomaly_scores()
        return self.evaluate_time_per_data, self.accuracy, self.f1

    def run_var(self, k, n_ohe, n_num, c_attack, c_normal):
        model_params = {
            'k': k,
            'n_ohe': n_ohe,
            'n_num': n_num,
            'c_attack': c_attack,
            'c_normal': c_normal,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_var(**model_params)
        self.fit()
        self.evaluate()
        self.print_results("Variance")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
        self.model.plot_anomaly_scores()
        return self.evaluate_time_per_data, self.accuracy, self.f1



