import pytest
import torch
import os
from src.preprocessing import preprocess_data
from src.app import SimpleCNN
from PIL import Image
import io


def test_preprocessing_logic(tmp_path):
    # Set up dummy raw data
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    os.makedirs(raw_dir / "cats")
    os.makedirs(raw_dir / "dogs")

    # Create 10 dummy images each
    for i in range(10):
        img = Image.new("RGB", (224, 224), color="red")
        img.save(raw_dir / "cats" / f"image_{i}.jpg")
        img.save(raw_dir / "dogs" / f"image_{i}.jpg")

    preprocess_data(str(raw_dir), str(processed_dir))

    # Check that at least some images went to train (80% of 10 = 8)
    assert len(os.listdir(processed_dir / "train" / "cats")) > 0
    assert len(os.listdir(processed_dir / "train" / "dogs")) > 0


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
    img = Image.new("RGB", (224, 224), color="red")
    # Use the same transform as in app.py
    from src.app import transform

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    assert probs.sum().item() == pytest.approx(1.0)
