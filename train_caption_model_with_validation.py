# train_caption_model_with_validation.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, LSTM, GRU, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import random

# Load data
with open('image_features.pkl', 'rb') as f:
    image_features = pickle.load(f)

from prepare_data_simple import load_captions, load_glove

captions = load_captions('Flicker8k_text/Flickr8k.token.txt')
embeddings_index = load_glove('glove/glove.6B.100d.txt')

# Tokenizer
all_captions = []
for caps in captions.values():
    all_captions.extend(caps)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

print(f'✅ Vocab size: {vocab_size}')
print(f'✅ Max caption length: {max_length}')

# Save max_length
with open('max_length.pkl', 'wb') as f:
    pickle.dump(max_length, f)
print('✅ Saved max_length.pkl')

# Embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Split captions into training and validation
image_names = list(captions.keys())
random.shuffle(image_names)

split_idx = int(0.8 * len(image_names))
train_image_names = image_names[:split_idx]
val_image_names = image_names[split_idx:]

train_captions = {img_name: captions[img_name] for img_name in train_image_names}
val_captions = {img_name: captions[img_name] for img_name in val_image_names}

print(f'✅ Training images: {len(train_captions)}, Validation images: {len(val_captions)}')

# Data generator
def data_generator(captions, image_features, tokenizer, max_length, vocab_size, batch_size=64):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for img_name, cap_list in captions.items():
            if img_name not in image_features:
                continue   # Skip missing image

            feature = image_features[img_name]
            for caption in cap_list:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield {'image_input': np.array(X1), 'seq_input': np.array(X2)}, np.array(y)
                        X1, X2, y = [], [], []
                        n = 0

# Model
def define_model(vocab_size, max_length, embedding_matrix, use_gru=False):
    # Image feature extractor
    inputs1 = Input(shape=(2048,), name='image_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence processor
    inputs2 = Input(shape=(max_length,), name='seq_input')
    se1 = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(inputs2)
    se2 = Dropout(0.5)(se1)
    
    # Use GRU or LSTM
    if use_gru:
        print("✅ Using GRU layer")
        se3 = GRU(256)(se2)
    else:
        print("✅ Using LSTM layer")
        se3 = LSTM(256)(se2)
    
    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# Build model (set use_gru=True if you want GRU)
model = define_model(vocab_size, max_length, embedding_matrix, use_gru=True)

# Print model summary
print(model.summary())

# Train model
epochs = 50
train_steps = len(train_captions) * 5 // 64   # heuristic
val_steps = len(val_captions) * 5 // 64       # heuristic

train_generator = data_generator(train_captions, image_features, tokenizer, max_length, vocab_size, batch_size=64)
val_generator = data_generator(val_captions, image_features, tokenizer, max_length, vocab_size, batch_size=64)

# Callbacks
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_image_caption_model.h5', monitor='val_loss', save_best_only=True)

# Fit model
model.fit(train_generator,
          validation_data=val_generator,
          epochs=epochs,
          steps_per_epoch=train_steps,
          validation_steps=val_steps,
          callbacks=[earlystop, checkpoint],
          verbose=1)

# Save final model
model.save('final_image_caption_model.h5')
print('✅ Final model saved as final_image_caption_model.h5')