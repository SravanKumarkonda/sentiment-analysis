from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import yaml
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.text_preprocessor import TextPreprocessor

class SentimentClassifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create pipeline with vectorizer and classifier
        self.model = Pipeline([
            ('vectorizer', TextPreprocessor(config_path).vectorizer),
            ('classifier', LogisticRegression(
                C=self.config['model']['classifier']['params']['C'],
                max_iter=self.config['model']['classifier']['params']['max_iter']
            ))
        ])
        
        self.logger = logging.getLogger(__name__)
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Create experiment if it doesn't exist
        experiment_name = self.config['mlflow']['experiment_name']
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            self.logger.error(f"Error setting up MLflow experiment: {str(e)}")
            raise
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the model and log metrics with MLflow"""
        with mlflow.start_run():
            try:
                # Train the pipeline
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred)
                }
                
                # Log parameters and metrics
                mlflow.log_params(self.config['model']['classifier']['params'])
                mlflow.log_metrics(metrics)
                
                # Log the pipeline
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name="sentiment_classifier"
                )
                
                self.logger.info(f"Training completed. Metrics: {metrics}")
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error in model training: {str(e)}")
                raise
    
    def predict(self, texts):
        """Make predictions on new data"""
        return self.model.predict(texts)