# prepare_data_simple.py

import os
import pickle
import numpy as np
from tqdm import tqdm

# Paths
CAPTIONS_PATH = "Flicker8k_text/Flickr8k.token.txt"
FEATURES_PATH = "image_features.pkl"
GLOVE_PATH = "glove/glove.6B.100d.txt"

# Load image features
with open(FEATURES_PATH, 'rb') as f:
    image_features = pickle.load(f)

print(f"✅ Loaded image features for {len(image_features)} images.")

# Load captions
def load_captions(captions_path):
    captions = {}
    with open(captions_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            img_name, caption = tokens
            img_name = img_name.split('#')[0]
            if img_name not in captions:
                captions[img_name] = []
            captions[img_name].append('start ' + caption.lower() + ' end')
    return captions

captions = load_captions(CAPTIONS_PATH)
print(f"✅ Loaded captions for {len(captions)} images.")

# Load GloVe embeddings
def load_glove(glove_path):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
    return embeddings_index

embeddings_index = load_glove(GLOVE_PATH)
print(f"✅ Loaded {len(embeddings_index)} word vectors from GloVe.")

# Simple verification
sample_img = list(captions.keys())[0]
print("\n📸 Sample image:", sample_img)
print("📝 Captions:")
for cap in captions[sample_img]:
    print("-", cap)