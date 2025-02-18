from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
            
        # Create TextPreprocessor instance (not just the vprint("Starting hyperparameter tuning with these options:")
        print(f"C values: {self.config['model']['classifier']['params']['C_options']}")
        print(f"max_iter values: {self.config['model']['classifier']['params']['max_iter_options']}")
        print(f"max_features values: {self.config['model']['vectorizer']['max_features_options']}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TextPreprocessor(config_path).vectorizer),
            ('classifier', LogisticRegression())
        ])
        
        # Define hyperparameter grid
        self.param_grid = {
            'classifier__C': self.config['model']['classifier']['params']['C_options'],
            'classifier__max_iter': self.config['model']['classifier']['params']['max_iter_options'],
            'vectorizer__max_features': self.config['model']['vectorizer']['max_features_options']
        }
        
        print("Hyperparameter grid:", self.param_grid)  # Add this debug print
        
        # Create GridSearchCV object
        self.model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=3
        )
        
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
        """Train the model with hyperparameter tuning and log metrics with MLflow"""
        with mlflow.start_run():
            try:
                print("Starting GridSearchCV with parameters:", self.param_grid)
                print("Number of parameter combinations:", len(self.model.cv_results_['params']) if hasattr(self.model, 'cv_results_') else "Not fitted yet")
                
                # Train with grid search
                self.model.fit(X_train, y_train)
                print("\nGrid search completed!")
                print("Best parameters:", self.model.best_params_)
                print("Best cross-validation score:", self.model.best_score_)
                
                # Log best parameters
                mlflow.log_params(self.model.best_params_)
                
                # Make predictions with best model
                y_pred = self.model.predict(X_test)
                
                # Calculate and log metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'best_cv_score': self.model.best_score_
                }
                
                mlflow.log_metrics(metrics)
                
                # Log all grid search results
                for i, (params, cv_score) in enumerate(zip(
                    self.model.cv_results_['params'],
                    self.model.cv_results_['mean_test_score']
                )):
                    mlflow.log_metric(f"trial_{i}_cv_score", cv_score)
                    for param_name, param_value in params.items():
                        mlflow.log_param(f"trial_{i}_{param_name}", param_value)
                
                # Log best model
                mlflow.sklearn.log_model(
                    self.model.best_estimator_,
                    "model",
                    registered_model_name="sentiment_classifier"
                )
                
                self.logger.info(f"Best parameters: {self.model.best_params_}")
                self.logger.info(f"Best CV score: {self.model.best_score_}")
                self.logger.info(f"Test metrics: {metrics}")
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error in model training: {str(e)}")
                raise

    def predict(self, texts):
        """Make predictions on new data"""
        return self.model.predict(texts)