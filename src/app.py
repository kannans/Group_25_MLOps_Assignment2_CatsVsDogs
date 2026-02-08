import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn
import logging
import time

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CatsDogsInference")

app = FastAPI(title="Cats vs Dogs Binary Classifier")

# Re-define SimpleCNN for loading
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
MODEL_PATH = "models/model.pt"
model = SimpleCNN()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")
else:
    logger.warning(f"Model file {MODEL_PATH} not found. Predict endpoint will fail.")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Metrics placeholders
request_count = 0

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    start_time = time.time()
    request_count += 1
    
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        label = "dog" if prediction == 1 else "cat"
        
        latency = (time.time() - start_time) * 1000 # ms
        
        # Logging
        logger.info(f"Prediction: {label}, Confidence: {confidence:.4f}, Latency: {latency:.2f}ms")
        
        return {
            "prediction": label,
            "confidence": confidence,
            "probabilities": {
                "cat": probs[0][0].item(),
                "dog": probs[0][1].item()
            },
            "latency_ms": latency,
            "request_count": request_count
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
