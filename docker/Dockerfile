FROM python:3.9-slim

WORKDIR /app

# Create cache directory for datasets
RUN mkdir -p /root/.cache/huggingface/datasets

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose ports for API and MLflow
EXPOSE 8000
EXPOSE 5000

# Start MLflow server and API
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 & uvicorn src.api.app:app --host 0.0.0.0 --port 8000"]