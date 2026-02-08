# Cats vs Dogs Classification - MLOps Pipeline

An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) with experiment tracking, containerization, and CI/CD.

## Features
- **Data Versioning**: DVC for dataset management
- **Experiment Tracking**: MLflow for metrics and model versioning
- **Inference API**: FastAPI with `/health` and `/predict` endpoints
- **Containerization**: Docker and Docker Compose
- **Kubernetes**: Production-ready deployment manifests
- **CI/CD**: GitHub Actions for automated testing, training, and deployment
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
├── k8s/
│   ├── deployment.yaml # Kubernetes deployment
│   └── service.yaml    # Kubernetes service
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Deployment setup
└── README.md           # This file
```

## Video Demo

### 📹 Watch the Demo Video

**Direct Link:** https://drive.google.com/xxxxxx

The video demonstration covers:
- **Data Preprocessing**: Dataset overview and 80/10/10 split with augmentation
- **Model Training**: Training baseline CNN model with MLflow tracking
- **MLflow Tracking**: Experiment tracking and model versioning
- **API Development**: FastAPI implementation with prediction endpoints
- **Docker Deployment**: Containerization and Docker setup
- **Kubernetes Deployment**: K8s deployment with services
- **CI/CD Pipeline**: GitHub Actions workflow demonstration
- **End-to-End Workflow**: Complete prediction pipeline demonstration

## Prerequisites
- Python 3.9+
- Git
- Docker & Docker Compose

## Installation

### Windows
```bash
# Clone repository
git clone https://github.com/kannans/Group_25_MLOps_Assignment2_CatsVsDogs.git
cd Group_25_MLOps_Assignment2_CatsVsDogs

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
git clone https://github.com/kannans/Group_25_MLOps_Assignment2_CatsVsDogs.git
cd Group_25_MLOps_Assignment2_CatsVsDogs

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
git clone https://github.com/kannans/Group_25_MLOps_Assignment2_CatsVsDogs.git
cd Group_25_MLOps_Assignment2_CatsVsDogs

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

**Note:** For Kubernetes deployment instructions, see [k8s/README.md](k8s/README.md) which includes Docker Desktop Kubernetes and Minikube setup for all platforms.

### Windows: Install Docker Desktop
1. Download Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/
2. Install with default options
3. Enable WSL 2 when prompted (if using WSL)
4. Restart your computer
5. Open Docker Desktop and wait until it shows "Docker Desktop is running"

Verify Docker Installation:
```bash
docker --version
docker info
```

### macOS: Install Docker Desktop
1. Download Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/
2. Choose the correct version:
   - Apple Silicon (M1/M2/M3): Download "Mac with Apple chip"
   - Intel Mac: Download "Mac with Intel chip"
3. Open the downloaded .dmg file
4. Drag Docker to Applications folder
5. Open Docker from Applications
6. Complete the setup wizard
7. Wait until Docker Desktop shows "Docker Desktop is running"

Verify Docker Installation:
```bash
docker --version
docker info
```

### Linux: Install Docker
Ubuntu/Debian:
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
```

Verify Docker Installation:
```bash
docker --version
docker info
```

### Build Docker Image
```bash
docker build -t cats-dogs-classifier .
```

### Run Container
```bash
docker run -p 8000:8000 cats-dogs-classifier
```

### Test Container
```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

### Using Docker Compose
```bash
docker-compose up --build
```

## Kubernetes Deployment

For complete Kubernetes deployment instructions, including setup for Windows, macOS, and Linux, see [k8s/README.md](k8s/README.md).

The Kubernetes deployment includes:
- Setup instructions for Docker Desktop Kubernetes, Minikube, and kind
- Step-by-step deployment guide
- Access methods (NodePort, Port Forward)
- Scaling and update procedures
- Complete troubleshooting guide

**Quick Start:**
```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Access
kubectl port-forward service/cats-dogs-classifier 8000:8000
```

## CI/CD Pipeline
GitHub Actions automatically:
- Runs linting (Black, Flake8) on every push
- Runs unit tests with coverage reporting
- Trains model and uploads artifacts
- Builds Docker image
- Deploys on merge to main branch

## Monitoring
- **Logging**: All requests logged with timestamp, prediction, and confidence
- **Metrics**: Request count and latency tracked per prediction
