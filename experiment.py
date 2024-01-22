import time
import plotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_noFS import AnomalyDetector_noFS
from model_hybrid import AnomalyDetector_hybrid
from model_var import AnomalyDetector_var
from model_PCA import AnomalyDetector_PCA
from model_FI import AnomalyDetector_FI
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

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train) # X_train: Pandas DataFrame, y_train: NumPy Array 
        self.fit_time = time.perf_counter() - start_time 

    def evaluate(self):
        start_time = time.perf_counter()
        self.prediction = self.model.predict(self.X_test) # X_test: Panda DataFrame
        self.evaluate_time = time.perf_counter() - start_time

    def print_results(self, title):
        self.accuracy, attack_acu, normal_acu = (round(metric, 2) for metric in 
                                            [accuracy_score(self.y_test, pred) for pred in 
                                            [self.prediction, self.model.attack_prd, self.model.normal_prd]])
        self.f1, f1_attack, f1_normal = (round(metric, 2) for metric in 
                                    [f1_score(self.y_test, pred, average='weighted') for pred in 
                                    [self.prediction, self.model.attack_prd, self.model.normal_prd]])
        fit_time, evaluate_time = round(self.fit_time, 2), round(self.evaluate_time, 2)
        print("------------------------------")
        print(f"Feature selection: {title} k:{self.model.k}")
        print(f"Accuracy: {self.accuracy}")
        print(f"Attack Subsystem Accuracy: {attack_acu}")
        print(f"Normal Subsystem Accuracy: {normal_acu}")
        print(f"F1 Score: {self.f1}")
        print(f"F1 Score Attack: {f1_attack}")
        print(f"F1 Score Normal: {f1_normal}")
        print(f"Fit Time: {fit_time}s")
        print(f"Evaluate Time: {evaluate_time}s")
        print("------------------------------")

    def run_noFS(self, k):
        model_params = {
            'k': k,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_noFS(**model_params)
        self.fit()
        self.evaluate()
        self.print_results("noFS")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)

    def run_pca(self, k, n_pca):
        model_params = {
            'k': k,
            'n_pca': n_pca,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_PCA(**model_params)
        self.fit()
        self.evaluate()
        self.print_results(f"PCA, n_pca={n_pca}")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)

    def run_fi(self, k, n_fi):
        model_params = {
            'k': k,
            'n_fi': n_fi,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_FI(**model_params)
        self.fit()
        self.evaluate()
        self.print_results(f"FI, n_fi={n_fi}")
        #plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)

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
        self.print_results(f"hybrid, n_fi={n_fi}, n_pca={n_pca}")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
    
    def run_var(self, k):
        model_params = {
            'k': k,
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector_var(**model_params)
        self.fit()
        self.evaluate()
        self.print_results("Variance")
        plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)



