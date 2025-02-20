from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import yaml
import logging
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.text_preprocessor import TextPreprocessor

class SentimentClassifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Create TextPreprocessor instance
        text_preprocessor = TextPreprocessor(config_path)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', text_preprocessor),
            ('classifier', LogisticRegression())
        ])
        
        # Define hyperparameter grid
        self.param_grid = {
            'classifier__C': self.config['model']['classifier']['params']['C_options'],
            'classifier__max_iter': self.config['model']['classifier']['params']['max_iter_options'],
            'vectorizer__max_features': self.config['model']['vectorizer']['max_features_options']
        }
        
        print("Hyperparameter grid:", self.param_grid)
        
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
                
                try:
                    # Get absolute path
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(current_dir))
                    results_path = os.path.join(project_root, 'src', 'results', 'model_results.json')
                    
                    print(f"Attempting to save results to: {results_path}")
                    
                    # Create directory if it doesn't exist
                    results_dir = os.path.dirname(results_path)
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Save results to JSON
                    results = {
                        'best_params': self.model.best_params_,
                        'metrics': metrics,
                        'all_results': {
                            'params': [dict(p) for p in self.model.cv_results_['params']],
                            'scores': self.model.cv_results_['mean_test_score'].tolist()
                        }
                    }
                    
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    
                    print(f"Successfully saved results to {results_path}")
                    self.logger.info(f"Saved results to {results_path}")
                    
                except Exception as e:
                    print(f"Error saving results: {str(e)}")
                    self.logger.error(f"Error saving results: {str(e)}")
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error in model training: {str(e)}")
                raise

    def predict(self, texts):
        """Make predictions on new data"""
        return self.model.predict(texts)