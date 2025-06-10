# extract_features.py

import os
import pickle
from tqdm import tqdm
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Folder where images are stored
IMAGES_PATH = 'Flicker8k_Dataset'

# Load ResNet50 model, remove final classification layer
model = ResNet50(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Function to load and preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Extract features for all images
features = {}
image_files = os.listdir(IMAGES_PATH)
total = len(image_files)

print(f"📸 Found {total} images in '{IMAGES_PATH}' folder.")

for img_name in tqdm(image_files):
    img_path = os.path.join(IMAGES_PATH, img_name)
    img = preprocess_image(img_path)
    feature = model.predict(img, verbose=0)
    feature = np.reshape(feature, feature.shape[1])
    features[img_name] = feature

# Save features to pickle file
with open('image_features.pkl', 'wb') as f:
    pickle.dump(features, f)

print('✅ Saved image_features.pkl')