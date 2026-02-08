# Cats vs Dogs Binary Classification - MLOps Assignment 2

An end-to-end MLOps pipeline for model building, artifact creation, packaging, containerization, and CI/CD-based deployment for a binary image classification task (Cats vs Dogs).

## Project Structure
```
├── data/               # Dataset files (managed by DVC)
│   ├── raw/            # Raw images
│   └── processed/      # Pre-processed images (224x224)
├── models/             # Trained model artifacts
├── notebooks/          # Exploratory Data Analysis
├── src/                # Source code
│   ├── __init__.py
│   ├── preprocessing.py # Data loading and augmentation
│   ├── train.py        # Model training and MLflow tracking
│   └── app.py          # FastAPI application
├── tests/              # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py # Tests for preprocessing and inference
├── .github/workflows/  # CI/CD pipelines
├── Dockerfile          # Containerization script
├── docker-compose.yml  # Local deployment setup
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Features
- **Data Versioning**: DVC for tracking datasets and pre-processed data.
- **Experiment Tracking**: MLflow for logging runs, metrics, and models.
- **Inference Service**: FastAPI with `/health` and `/predict` endpoints.
- **Containerization**: Docker for reproducible environments.
- **CI/CD**: GitHub Actions for automated testing and image building.
- **Testing**: Unit tests with `pytest`.

## Prerequisites
- Python 3.9+
- Git
- Docker & Docker Compose

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-assignment2
```

### 2. Create Virtual Environment
#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC
```bash
dvc init
```

## Running the Project

### 1. Data Preprocessing
Pre-processes images to 224x224 RGB and splits into 80% train, 10% validation, and 10% test sets with data augmentation.
```bash
python -m src.preprocessing
```

### 2. Model Training & Experiment Tracking
Trains a baseline CNN model and logs parameters, metrics (confusion matrix, loss curves), and artifacts to MLflow.
```bash
python -m src.train
```
View MLflow UI:
```bash
mlflow ui
```

### 3. Run Inference Service Locally
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- **Health Check**: `GET http://localhost:8000/health`
- **Predict**: `POST http://localhost:8000/predict` (Accepts image file)

### 4. Run Tests
```bash
pytest tests/ -v
```

## Docker Deployment
### Build Docker Image
```bash
docker build -t cats-dogs-classifier .
```
### Run Container
```bash
docker run -p 8000:8000 cats-dogs-classifier
```
### Using Docker Compose
```bash
docker-compose up --build
```

## CI/CD Pipeline
- **Continuous Integration**: Automated testing and Docker image building via GitHub Actions.
- **Continuous Deployment**: Auto-deployment to target environment upon push to main branch.

## Monitoring & Logging
- **Logging**: Request/Response logging enabled in the inference service.
- **Metrics**: Tracks request count and latency.
