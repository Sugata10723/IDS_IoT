import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, data, target):
        self.data = data
        self.target = target.values.ravel()
        self.model_instance = model

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )

    def fit(self):
        start_time = time.perf_counter()
        self.model_instance.fit(self.X_train, self.y_train)
        self.time = time.perf_counter() - start_time

    def evaluate(self):
        y_pred = self.model_instance.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        self.f1 = f1_score(self.y_test, y_pred, average='weighted')

    def get_metrics(self):
        return {"accuracy": self.accuracy, "f1_score": self.f1, "time": self.time}
