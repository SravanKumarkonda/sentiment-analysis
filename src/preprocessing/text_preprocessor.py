from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging
import mlflow
import pickle
import os

class TextPreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['model']['vectorizer']['max_features'],
            min_df=self.config['model']['vectorizer']['min_df'],
            max_df=self.config['model']['vectorizer']['max_df']
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_params(self, deep=True):
        """Get parameters for GridSearchCV compatibility"""
        return {
            'max_features': self.vectorizer.max_features,
            'min_df': self.vectorizer.min_df,
            'max_df': self.vectorizer.max_df
        }
    
    def set_params(self, **parameters):
        """Set parameters for GridSearchCV compatibility"""
        for parameter, value in parameters.items():
            setattr(self.vectorizer, parameter, value)
        return self
    
    def fit_transform(self, texts):
        """Fit and transform the text data"""
        try:
            vectors = self.vectorizer.fit_transform(texts)
            # Save vectorizer alongside the model
            if mlflow.active_run():
                # Save as part of the model artifacts
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