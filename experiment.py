import mlflow
import mlflow.sklearn
from preprocessor import Preprocessor
from trainer import Trainer
from ids import IoTAnomalyDetector

class Experiment:
    def __init__(self, data, labels, config):
        self.data = data
        self.labels = labels
        self.config = config

    def _run_experiment(self, model_cls=None, model_params=None):
        
        # Ensure we have parameters to pass to the model
        model_params = model_params or {}

        with mlflow.start_run():
            processed_data = self._preprocess_data()
            
            # Initialize and train the model
            model_instance = model_cls(**model_params)
            trainer = Trainer(model_instance, processed_data, self.labels)
            trainer.split_data()
            trainer.fit()
            trainer.evaluate()

            # Log the model and its metrics to MLflow
            mlflow.log_params(model_params)

            self._log_metrics_and_model(trainer, model_instance)

    def _preprocess_data(self):
        pps = Preprocessor(self.data, self.config)
        pps.process()
        return pps.get_processed_data()

    def _log_metrics_and_model(self, trainer, model_instance):
        for key, value in trainer.get_metrics().items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(model_instance, "model")

