# lip_reader.py
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load extracted features
with open("dataset/processed_lip_features.json", "r") as f:
    processed_data = json.load(f)

# Prepare training data
X = []
y = []
for label, sequences in processed_data.items():
    for seq in sequences:
        X.append([*seq["centroid"], *seq["bounding_box"]])
        y.append(label)

X = np.array(X)
y = np.array(y)

# Feature normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Save the trained model and scaler
import joblib
joblib.dump(knn, "lip_reader_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training completed.")
