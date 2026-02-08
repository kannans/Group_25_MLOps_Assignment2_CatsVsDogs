import os

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# We would normally download the dataset from Kaggle here if possible.
# For now, we assume the user will place raw data in data/raw
print("Environment prepared. Please place the Cats and Dogs dataset in data/raw.")
