# Customer Churn Prediction (Production-ready ML Project)

## Features

- Data loading, EDA, feature engineering
- ML pipeline (scikit-learn)
- Model export with joblib
- Flask API
- Docker containerization
- Airflow for automated data loading
- GitHub Actions CI/CD → AWS ECS deployment

## Commands

```bash
# Train model
python app/train.py

# Run Flask app
python app/flask_app.py

# Build Docker image
docker build -t churn-app -f docker/Dockerfile .

# Run Docker container
docker run -p 5000:5000 churn-app
