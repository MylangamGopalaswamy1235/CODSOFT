# prepare_data.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle
from tqdm import tqdm

# Image path
IMAGES_PATH = 'Flicker8k_Dataset/'

# Load ResNet50 with fine-tuning
resnet = ResNet50(weights='imagenet', include_top=False, pooling=None)  # No GlobalPooling → get (7, 7, 2048)

# Freeze all layers first
for layer in resnet.layers:
    layer.trainable = False

# Unfreeze last 10 layers
for layer in resnet.layers[-10:]:
    layer.trainable = True

print("✅ ResNet50 loaded with last 10 layers trainable.")

# Define model that outputs feature map (7x7x2048)
model = tf.keras.Model(resnet.input, resnet.output)

# Extract features
features = {}
total = len(os.listdir(IMAGES_PATH))
for i, img_name in enumerate(tqdm(os.listdir(IMAGES_PATH), desc="Extracting features")):
    filename = IMAGES_PATH + img_name
    try:
        img = image.load_img(filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feature_map = model.predict(img, verbose=0)  # shape: (1, 7, 7, 2048)
        feature_map = np.reshape(feature_map, (49, 2048))  # (49, 2048)

        features[img_name] = feature_map
    except:
        print(f"⚠ Skipping image {img_name}")

# Save features
with open('image_features_attention.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"✅ Extracted and saved features for {len(features)} images to 'image_features_attention.pkl'")