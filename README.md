# Cats vs Dogs Classification - MLOps Pipeline

An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) with experiment tracking, containerization, and CI/CD.

## Features
- **Data Versioning**: DVC for dataset management
- **Experiment Tracking**: MLflow for metrics and model versioning
- **Inference API**: FastAPI with `/health` and `/predict` endpoints
- **Containerization**: Docker and Docker Compose
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Request/response logging and latency tracking

## Project Structure
```
├── data/               # Dataset (managed by DVC)
├── models/             # Trained models (.pt files)
├── src/
│   ├── preprocessing.py # Data preprocessing (80/10/10 split + augmentation)
│   ├── train.py        # Model training with MLflow
│   └── app.py          # FastAPI inference service
├── tests/              # Unit tests and smoke tests
├── .github/workflows/  # CI/CD pipelines
├── Dockerfile          # Container configuration
└── docker-compose.yml  # Deployment setup
```

## Prerequisites
- Python 3.9+
- Git
- Docker & Docker Compose

## Installation

### Windows
```bash
# Clone repository
git clone <repository-url>
cd mlops-assignment2

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS
```bash
# Clone repository
git clone <repository-url>
cd mlops-assignment2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Ubuntu/Linux
```bash
# Clone repository
git clone <repository-url>
cd mlops-assignment2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

### 1. Data Preprocessing
```bash
python -m src.preprocessing
```

### 2. Train Model
```bash
python -m src.train
```

View MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

### 3. Run Inference Service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

API Endpoints:
- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict` (upload image file)

### 4. Test Prediction
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

### 5. Run Tests
```bash
pytest tests/ -v
```

## Docker Deployment

### Build and Run
```bash
docker build -t cats-dogs-classifier .
docker run -p 8000:8000 cats-dogs-classifier
```

### Using Docker Compose
```bash
docker-compose up --build
```

### Smoke Test
```bash
python tests/smoke_test.py
```

## Kubernetes Deployment

### Prerequisites
- kubectl
- Local cluster: minikube, kind, or microk8s

### Deploy
```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# For minikube
minikube start
eval $(minikube docker-env)
docker build -t cats-dogs-classifier:latest .

# For kind
kind create cluster --name mlops-cluster
kind load docker-image cats-dogs-classifier:latest --name mlops-cluster

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Access service
kubectl port-forward service/cats-dogs-classifier 8000:8000
```

See [k8s/README.md](k8s/README.md) for detailed instructions.

## CI/CD Pipeline
GitHub Actions automatically:
- Runs unit tests on every push
- Builds Docker image
- Deploys on merge to main branch

## Monitoring
- **Logging**: All requests logged with timestamp, prediction, and confidence
- **Metrics**: Request count and latency tracked per prediction
