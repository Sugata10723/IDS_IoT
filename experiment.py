import time
from model import AnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Experiment:
    def __init__(self, data, labels, config):
        self.data = data # Pandas DataFrame
        self.labels = labels # Pandas Series
        self.config = config
        self.model = None
        self.X_train = None # Pandas DataFrame
        self.X_test = None # Pandas DataFrame
        self.y_train = None # Pandas Series
        self.y_test = None # Pandas Series
        self.accuracy = None
        self.f1 = None
        self.fit_time = None
        self.evaluate_time = None

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )
        self.y_train = self.y_train.values # Pandas Series -> NumPy Array
        self.y_test = self.y_test.values # Pandas Series -> NumPy Array

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train) # X_train: Pandas DataFrame, y_train: NumPy Array 
        self.fit_time = time.perf_counter() - start_time 

    def evaluate(self):
        start_time = time.perf_counter()
        y_pred = self.model.predict(self.X_test) # X_test: Panda DataFrame
        self.evaluate_time = time.perf_counter() - start_time
        
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred, average='weighted') # 2値分類問題だが、y_predに-1が含まれるためweightedを使用

    def print_results(self):
        print('Fit time: {:.4f}'.format(self.fit_time))
        print('Evaluate time: {:.4f}'.format(self.evaluate_time))
        print('Accuracy: {:.4f}'.format(self.accuracy))
        print('F1: {:.4f}'.format(self.f1))

    def run(self):
        model_params = {
            'k': self.config['k'],
            'categorical_columns': self.config['categorical_columns']
        }
        self.model = AnomalyDetector(**model_params)
        self.split_data()
        self.fit()
        self.evaluate()
        self.print_results()




