# extract_features_simple.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
import pickle
from tqdm import tqdm

# Paths
IMAGES_PATH = "Flicker8k_Dataset"
FEATURES_PATH = "image_features.pkl"

# Load ResNet50 without last layer
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Extract features from a single image
def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x, verbose=0)
    feature = np.reshape(feature, feature.shape[1])
    return feature

# Process all images
print("📷 Extracting features for all images...")
features = {}

all_images = os.listdir(IMAGES_PATH)
total = len(all_images)

for img_name in tqdm(all_images):
    img_path = os.path.join(IMAGES_PATH, img_name)
    feature = extract_feature(img_path)
    features[img_name] = feature

# Save features to disk
print("💾 Saving extracted features...")
with open(FEATURES_PATH, 'wb') as f:
    pickle.dump(features, f)

print(f"✅ Done! Extracted features for {total} images. Saved to {FEATURES_PATH}.")