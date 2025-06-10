import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# --- CONFIGURATION ---
IMAGE_NAME = "44856031_0d82c2c7d1.jpg"   # your given image
MODEL_PATH = "image_caption_model.h5"   # your model file
FEATURES_PATH = "image_features.pkl"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH = 40  # max caption length

# --- LOAD TOKENIZER ---
print("📥 Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

word_to_index = tokenizer.word_index
index_to_word = {v: k for k, v in word_to_index.items()}
vocab_size = len(tokenizer.word_index) + 1

# --- LOAD IMAGE FEATURES ---
print("📥 Loading image features...")
with open(FEATURES_PATH, 'rb') as f:
    image_features = pickle.load(f)

# --- LOAD MODEL ---
print(f"📥 Loading trained model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# --- GENERATE CAPTION FUNCTION ---
def generate_caption(model, tokenizer, image_feature, max_length):
    in_text = 'startseq'
    prev_word = None
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([np.array([image_feature]), sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = index_to_word.get(yhat_index, None)

        # Stop if no word or stop token predicted
        if word is None or word in ['endseq', 'end']:
            break

        # Stop if immediate repetition detected
        if word == prev_word:
            break

        in_text += ' ' + word
        prev_word = word

    # Clean up start and end tokens
    final_caption = in_text.replace('startseq', '').strip()
    final_words = final_caption.split()
    while final_words and final_words[-1] in ['endseq', 'end']:
        final_words.pop()
    final_caption = ' '.join(final_words)
    return final_caption

# --- SELECT IMAGE FEATURE ---
print(f"\n🎯 Selecting image: {IMAGE_NAME}")
if IMAGE_NAME not in image_features:
    raise ValueError(f"❌ Image '{IMAGE_NAME}' not found in image_features.pkl!")

test_img_feature = image_features[IMAGE_NAME]

# --- GENERATE AND PRINT CAPTION ---
print("📝 Generating caption...")
caption = generate_caption(model, tokenizer, test_img_feature, MAX_LENGTH)
print(f"\n🖼 Image: {IMAGE_NAME}")
print(f"📝 Caption: \"{caption}\"")