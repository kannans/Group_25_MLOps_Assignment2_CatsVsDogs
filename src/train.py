import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Simple CNN Baseline Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # Assuming 224x224 input
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(data_dir, epochs=5, lr=0.001, batch_size=32):
    # MLflow Setup
    mlflow.set_experiment("Cats_vs_Dogs_Classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_type", "SimpleCNN")

        # Data Loading
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} not found. Skipping training.")
            return

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save Model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        print(f"Model saved to {model_path} and logged to MLflow.")

if __name__ == "__main__":
    # Assuming data layout is data/processed/train and data/processed/val
    train_model("data/processed", epochs=2)
