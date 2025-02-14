from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import yaml
import logging
from typing import List
from src.preprocessing.text_preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Setup MLflow
mlflow.set_tracking_uri("http://sentiment-mlflow:5000")

class TextInput(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
async def predict(input_data: TextInput):
    """Make predictions on new text data"""
    try:
        # Load the pipeline from MLflow
        model = mlflow.sklearn.load_model("models:/sentiment_classifier/latest")
        logger.info("Successfully loaded model")
        
        # Make predictions using the pipeline
        predictions = model.predict(input_data.texts)
        
        return {
            "predictions": predictions.tolist(),
            "sentiment": ["Positive" if p == 1 else "Negative" for p in predictions]
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = "sentiment_classifier"
        
        latest_version = client.get_latest_versions(model_name, stages=["None"])
        if not latest_version:
            raise HTTPException(status_code=404, detail="No model found")
            
        model_version = latest_version[0]
        
        return {
            "name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "status": model_version.status
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))