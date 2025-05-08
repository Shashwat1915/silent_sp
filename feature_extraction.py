# feature_extraction.py
import numpy as np
import json

# Function to compute centroid of coordinates
def compute_centroid(coords):
    x_vals = [x for x, y in coords]
    y_vals = [y for x, y in coords]
    return np.mean(x_vals), np.mean(y_vals)

# Function to compute bounding box
def compute_bounding_box(coords):
    x_vals = [x for x, y in coords]
    y_vals = [y for x, y in coords]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    return (min_x, min_y, max_x, max_y)

# Load the dataset
DATASET_FILE = "dataset/lip_dataset.json"
with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

# Process the dataset to extract features
processed_data = {}
for label, sequences in dataset.items():
    processed_data[label] = []
    for seq in sequences:
        centroid = compute_centroid(seq)
        bounding_box = compute_bounding_box(seq)
        processed_data[label].append({"centroid": centroid, "bounding_box": bounding_box})

# Save processed features
with open("dataset/processed_lip_features.json", "w") as f:
    json.dump(processed_data, f)

print("✅ Feature extraction completed.")
