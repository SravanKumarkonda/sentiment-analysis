from datasets import load_dataset
from sklearn.model_selection import train_test_split
import yaml
import logging

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load IMDB dataset and split into train/test"""
        try:
            # Load dataset with specific configuration
            dataset = load_dataset(
                "imdb",
                download_mode="force_redownload",
                verification_mode="no_checks"
            )
            train_data = dataset['train']
            
            texts = train_data['text']
            labels = train_data['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                texts, 
                labels,
                test_size=1-self.config['data']['train_size'],
                random_state=self.config['data']['random_state']
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise