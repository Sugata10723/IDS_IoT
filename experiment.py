import time
import plotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_noFS import AnomalyDetector_noFS
from model_hybrid import AnomalyDetector_hybrid
from model_var import AnomalyDetector_var
from model_mean import AnomalyDetector_mean
from model_cor import AnomalyDetector_cor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class Experiment:
    def __init__(self, dataset):
        self.X_train = dataset[0] # Pandas DataFrame
        self.X_test = dataset[1] # Pandas DataFrame
        self.y_train = dataset[2] # ndarray
        self.y_test = dataset[3] # ndarray
        self.config = dataset[4]
        self.model = None
        self.accuracy = None
        self.f1 = None
        self.fit_time = None
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
        evaluate_time = time.perf_counter() - start_time
        
        self.accuracy = accuracy_score(self.y_test, self.prediction)
        self.f1 = f1_score(self.y_test, self.prediction, average='weighted')
        self.evaluate_time_per_data = evaluate_time / len(self.y_test) * 1e6

    def print_results(self, title):
        print("------------------------------")
        print(f"Feature selection: {title} k:{self.model.k}")
        print(f"Accuracy: {format(self.accuracy, '.3f')}")
        print(f"F1 Score: {format(self.f1, '.3f')}")
        print(f"Fit Time: {format(self.fit_time, '.3f')}s")
        print(f"Evaluate Time per Data: {format(self.evaluate_time_per_data, '.3f')}us")
        print("------------------------------")

    def run_noFS(self, model_params, if_plot):
        self.model = AnomalyDetector_noFS(model_params, self.config['categorical_columns'])
        self.fit()
        self.evaluate()
        if if_plot:
            self.print_results("noFS")
            plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
            self.model.plot_anomaly_scores()

    def run_mean(self, model_params, if_plot):
        self.model = AnomalyDetector_mean(model_params, self.config['categorical_columns'])
        self.fit()
        self.evaluate()
        if if_plot:
            self.print_results("mean")
            plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
            self.model.plot_anomaly_scores()

    def run_hybrid(self, model_params, if_plot):
        self.model = AnomalyDetector_hybrid(model_params, self.config['categorical_columns'])
        self.fit()
        self.evaluate()
        if if_plot:
            self.print_results(f"hybrid, n_fi={model_params['n_fi']}, n_pca={model_params['n_pca']}")
            plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
            self.model.plot_anomaly_scores()

    def run_var(self, model_params, if_plot):
        self.model = AnomalyDetector_var(model_params, self.config['categorical_columns'])
        self.fit()
        self.evaluate()
        if if_plot:
            self.print_results("Variance")
            plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
            self.model.plot_anomaly_scores()

    def run_cor(self, model_params, if_plot):
        self.model = AnomalyDetector_cor(model_params, self.config['categorical_columns'])
        self.fit()
        self.evaluate()
        if if_plot:
            self.print_results("Correlation")
            plotter.plot_confusion_matrix(self.y_test, self.prediction, self.model.attack_prd, self.model.normal_prd)
            self.model.plot_anomaly_scores()



