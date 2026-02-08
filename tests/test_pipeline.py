import pytest
import torch
import os
from src.preprocessing import preprocess_data
from src.app import SimpleCNN
from PIL import Image
import io

def test_preprocessing_indices():
    # Mock data_dir check
    if not os.path.exists("data/raw"):
        pytest.skip("Raw data directory not found")
    
    train_idx, val_idx, test_idx = preprocess_data("data/raw")
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

def test_model_output_shape():
    model = SimpleCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2)

def test_inference_logic():
    model = SimpleCNN()
    model.eval()
    # Dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    # Use the same transform as in app.py
    from src.app import transform
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    assert probs.sum().item() == pytest.approx(1.0)
