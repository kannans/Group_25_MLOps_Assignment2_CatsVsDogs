# Kubernetes Deployment Guide

## Prerequisites
- Docker
- kubectl
- One of: kind, minikube, or microk8s

## Setup Local Kubernetes Cluster

### Using Minikube
```bash
# Start minikube
minikube start

# Enable ingress (optional)
minikube addons enable ingress

# Use minikube's Docker daemon
eval $(minikube docker-env)
```

### Using kind
```bash
# Create cluster
kind create cluster --name mlops-cluster

# Load image into kind
kind load docker-image cats-dogs-classifier:latest --name mlops-cluster
```

### Using microk8s
```bash
# Install microk8s (Ubuntu)
sudo snap install microk8s --classic

# Enable required addons
microk8s enable dns storage
```

## Build and Load Docker Image

```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# For minikube (already using minikube docker-env)
# Image is already available

# For kind
kind load docker-image cats-dogs-classifier:latest --name mlops-cluster

# For microk8s
docker save cats-dogs-classifier:latest | microk8s ctr image import -
```

## Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -l app=cats-dogs-classifier
```

## Access the Service

### Minikube
```bash
# Get service URL
minikube service cats-dogs-classifier --url

# Or use port-forward
kubectl port-forward service/cats-dogs-classifier 8000:8000
```

### kind
```bash
# Port forward
kubectl port-forward service/cats-dogs-classifier 8000:8000
```

### microk8s
```bash
# Port forward
microk8s kubectl port-forward service/cats-dogs-classifier 8000:8000
```

## Test the Deployment

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

## Cleanup

```bash
# Delete resources
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml

# Stop cluster
minikube stop  # for minikube
kind delete cluster --name mlops-cluster  # for kind
microk8s stop  # for microk8s
```
