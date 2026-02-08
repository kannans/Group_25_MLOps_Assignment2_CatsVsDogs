# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
# Install CPU-only PyTorch separately and then install the rest of the requirements, 
# ensuring torch/torchvision aren't re-installed by removing them from the requirements file in-container.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && \
    sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/
COPY README.md .

# Create models directory (assuming model is built during CI or mounted)
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
