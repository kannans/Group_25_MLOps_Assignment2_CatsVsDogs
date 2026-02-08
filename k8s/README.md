# Kubernetes Deployment Guide

This guide covers deploying the Cats vs Dogs classifier to a local Kubernetes cluster.

## Prerequisites
- Docker
- kubectl
- One of: Docker Desktop with Kubernetes, Minikube, or kind

## Setup Local Kubernetes Cluster

### Option 1: Docker Desktop Kubernetes (Recommended for Windows/macOS)

#### Windows
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Open Docker Desktop → Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Verify: `kubectl cluster-info`

#### macOS
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Open Docker Desktop → Preferences → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Verify: `kubectl cluster-info`

### Option 2: Minikube

#### Windows
```bash
# Install minikube
choco install minikube

# Start cluster
minikube start

# Verify
kubectl get nodes
```

#### macOS
```bash
# Install minikube
brew install minikube

# Start cluster
minikube start

# Verify
kubectl get nodes
```

#### Linux
```bash
# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster
minikube start

# Verify
kubectl get nodes
```

### Option 3: kind (Kubernetes in Docker)

```bash
# Install kind
# macOS
brew install kind

# Windows (using chocolatey)
choco install kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create cluster
kind create cluster --name mlops-cluster

# Verify
kubectl cluster-info --context kind-mlops-cluster
```

## Build and Load Docker Image

### For Docker Desktop Kubernetes
```bash
# Build image (will be available to k8s automatically)
docker build -t cats-dogs-classifier:latest .
```

### For Minikube
```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build image
docker build -t cats-dogs-classifier:latest .
```

### For kind
```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Load into kind cluster
kind load docker-image cats-dogs-classifier:latest --name mlops-cluster
```

## Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Apply service
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Check pod logs
kubectl logs -l app=cats-dogs-classifier
```

## Access the Service

### Method 1: Port Forward (Works for all setups)
```bash
kubectl port-forward service/cats-dogs-classifier 8000:8000
```
Access at: `http://localhost:8000`

### Method 2: Minikube Service
```bash
# Get service URL
minikube service cats-dogs-classifier --url
```

### Method 3: Docker Desktop
The LoadBalancer service will be accessible at `http://localhost:8000`

## Test the Deployment

```bash
# Health check
curl http://localhost:8000/health

# Prediction (replace with actual image path)
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/cat_or_dog.jpg"
```

## Scaling

```bash
# Scale to 3 replicas
kubectl scale deployment cats-dogs-classifier --replicas=3

# Verify
kubectl get pods
```

## Update Deployment

```bash
# After rebuilding image
docker build -t cats-dogs-classifier:latest .

# For kind, reload image
kind load docker-image cats-dogs-classifier:latest --name mlops-cluster

# Restart deployment
kubectl rollout restart deployment cats-dogs-classifier

# Check rollout status
kubectl rollout status deployment cats-dogs-classifier
```

## Troubleshooting

### Pods not starting
```bash
# Check pod status
kubectl get pods

# View pod details
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### Image pull errors
- For Docker Desktop: Ensure image is built locally
- For Minikube: Use `eval $(minikube docker-env)` before building
- For kind: Use `kind load docker-image` after building

### Service not accessible
```bash
# Check service
kubectl get svc cats-dogs-classifier

# Verify endpoints
kubectl get endpoints cats-dogs-classifier

# Use port-forward as fallback
kubectl port-forward service/cats-dogs-classifier 8000:8000
```

## Cleanup

```bash
# Delete resources
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml

# Stop cluster
minikube stop  # for minikube
kind delete cluster --name mlops-cluster  # for kind
# Docker Desktop: Disable Kubernetes in settings
```

## Monitoring

```bash
# Watch pods
kubectl get pods -w

# View resource usage
kubectl top pods

# View events
kubectl get events --sort-by='.lastTimestamp'
```
