import logging
from src.data.data_loader import DataLoader
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_classifier import SentimentClassifier

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = "config/config.yaml"
        
        # Load and split data
        logger.info("Loading data...")
        data_loader = DataLoader(config_path)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        # Train model with MLflow
        logger.info("Training model...")
        classifier = SentimentClassifier(config_path)
        metrics = classifier.train(X_train, y_train, X_test, y_test)
        
        logger.info(f"Training completed. Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()