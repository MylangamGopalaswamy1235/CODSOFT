# train_caption_model_simple.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Dropout, add
from tensorflow.keras.models import Model
import pickle
import numpy as np

# Load prepared data
with open('image_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

# Simulate loading captions (use your real caption tokenizer in practice)
word_to_index = {'start': 1, 'a': 2, 'man': 3, 'riding': 4, 'horse': 5, 'end': 6}
index_to_word = {v: k for k, v in word_to_index.items()}
vocab_size = len(word_to_index) + 1
max_length = 6

# Dummy data generation for example (replace with real training data preparation)
def generate_dummy_data(image_features, num_samples=100):
    X1, X2, y = [], [], []
    for i, (img_name, feature) in enumerate(image_features.items()):
        if i >= num_samples:
            break
        X1.append(feature)
        X2.append([1, 2, 3, 4, 5, 6])  # Simulated sequence of word indexes
        y.append(tf.keras.utils.to_categorical([3], num_classes=vocab_size)[0])
    return np.array(X1), np.array(X2), np.array(y)

# Define model
def build_caption_model():
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = GRU(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Build model
print("🧠 Building captioning model...")
model = build_caption_model()

# Prepare dummy data
print("📦 Preparing dummy training data...")
X1_train, X2_train, y_train = generate_dummy_data(image_features)

# Train model
print("🚀 Starting training...")
model.fit([X1_train, X2_train], y_train, epochs=5, batch_size=16)

# Save model
print("💾 Saving trained model...")
model.save('caption_model_simple.h5')

print("✅ Done! Model saved as caption_model_simple.h5")