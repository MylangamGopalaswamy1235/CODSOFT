# generate_caption_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# --- CONFIGURATION ---
IMAGE_NAME = "667626_18933d713e.jpg"
MODEL_PATH = "image_caption_model_with_attention.h5"
FEATURES_PATH = "image_features_attention.pkl"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH_PATH = "max_length.pkl"
BEAM_SIZE = 3
REPETITION_PENALTY = 1.2  # stronger penalty

# --- LOAD TOKENIZER ---
print("📥 Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

word_to_index = tokenizer.word_index
index_to_word = {v: k for k, v in word_to_index.items()}
vocab_size = len(tokenizer.word_index) + 1

# --- LOAD MAX LENGTH ---
with open(MAX_LENGTH_PATH, 'rb') as f:
    max_length = pickle.load(f)

# --- LOAD IMAGE FEATURES ---
print("📥 Loading image features...")
with open(FEATURES_PATH, 'rb') as f:
    image_features = pickle.load(f)

# --- IMPORT CUSTOM LAYER ---
from train_caption_model_with_attention import AttentionLayer

# --- LOAD MODEL ---
print(f"📥 Loading trained model from {MODEL_PATH}...")
model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})

# --- GENERATE CAPTION BEAM SEARCH ---
def generate_caption_beam_search(model, tokenizer, image_feature, max_length, beam_size=3, repetition_penalty=1.2):
    start_seq = 'startseq'
    sequences = [[start_seq, 0.0]]

    while True:
        all_candidates = []
        for seq, score in sequences:
            sequence_encoded = tokenizer.texts_to_sequences([seq])[0]
            sequence_encoded = pad_sequences([sequence_encoded], maxlen=max_length)

            yhat = model.predict([np.array([image_feature]), sequence_encoded], verbose=0)
            yhat = np.log(yhat[0] + 1e-9)

            top_indices = np.argsort(yhat)[-beam_size:]

            for idx in top_indices:
                word = index_to_word.get(idx, None)
                if word is None or word == 'startseq':
                    continue

                # Avoid adding multiple 'endseq'
                if word == 'endseq' and 'endseq' in seq:
                    continue

                candidate_seq = seq + ' ' + word
                candidate_score = score + yhat[idx]

                # Penalize repetition
                if word in seq.split():
                    candidate_score -= repetition_penalty

                all_candidates.append([candidate_seq, candidate_score])

        # Select top beam_size
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]

        # Stop if all sequences ended
        complete = True
        for seq, score in sequences:
            if not seq.endswith('endseq'):
                complete = False
                break
        if complete:
            break

        # Also stop if length too long
        if len(sequences[0][0].split()) >= max_length:
            break

    # Best sequence
    best_seq = sequences[0][0]
    final_caption = best_seq.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# --- SELECT IMAGE FEATURE ---
print(f"\n🎯 Selecting image: {IMAGE_NAME}")
if IMAGE_NAME not in image_features:
    raise ValueError(f"❌ Image '{IMAGE_NAME}' not found in image_features.pkl!")

test_img_feature = image_features[IMAGE_NAME]

# --- GENERATE AND PRINT CAPTION ---
print("📝 Generating caption with Beam Search...")
caption = generate_caption_beam_search(model, tokenizer, test_img_feature, max_length, beam_size=BEAM_SIZE, repetition_penalty=REPETITION_PENALTY)
print(f"\n🖼 Image: {IMAGE_NAME}")
print(f"📝 Caption: \"{caption}\"")