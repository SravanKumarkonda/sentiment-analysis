from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import yaml
import logging
import mlflow
import pickle
import os

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config_path: str = None, vectorizer__max_features=None):
        self.config_path = config_path  # Store config_path
        self.vectorizer__max_features = vectorizer__max_features  # Store max_features
        
        # Initialize vectorizer with default values
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Default value
            min_df=5,
            max_df=0.7
        )
        
        # If config path is provided, load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Update vectorizer with config values
            self.vectorizer.set_params(
                max_features=vectorizer__max_features or self.config['model']['vectorizer']['max_features'],
                min_df=self.config['model']['vectorizer']['min_df'],
                max_df=self.config['model']['vectorizer']['max_df']
            )
        
        self.logger = logging.getLogger(__name__)
    
    def get_params(self, deep=True):
        """Get parameters for GridSearchCV compatibility"""
        return {
            'config_path': self.config_path,
            'vectorizer__max_features': self.vectorizer.max_features
        }
    
    def set_params(self, **parameters):
        """Set parameters for GridSearchCV compatibility"""
        for parameter, value in parameters.items():
            if parameter == 'vectorizer__max_features':
                self.vectorizer.max_features = value
            elif parameter == 'config_path':
                self.config_path = value
        return self
    
    def fit_transform(self, texts, y=None):
        """Fit and transform the text data"""
        try:
            vectors = self.vectorizer.fit_transform(texts)
            if mlflow.active_run():
                with open("vectorizer.pkl", "wb") as f:
                    pickle.dump(self.vectorizer, f)
                mlflow.log_artifact("vectorizer.pkl", artifact_path="model")
            return vectors
        except Exception as e:
            self.logger.error(f"Error in text vectorization: {str(e)}")
            raise
    
    def transform(self, texts):
        """Transform new text data"""
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            self.logger.error(f"Error in text transformation: {str(e)}")
            raise