# train_caption_model_with_attention.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tqdm import tqdm

# Load data
with open('image_features_attention.pkl', 'rb') as f:
    image_features = pickle.load(f)

from prepare_data_simple import load_captions, load_glove  # reuse your existing prepare_data_simple.py
from attention_layer import AttentionLayer  # use attention_layer.py file separately!

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

# Save tokenizer & max_length
with open('max_length.pkl', 'wb') as f:
    pickle.dump(max_length, f)
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Data generator
def data_generator(captions, image_features, tokenizer, max_length, vocab_size, batch_size=64):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for img_name, cap_list in captions.items():
            if img_name not in image_features:
                continue

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

# Define model
def define_model(vocab_size, max_length, embedding_matrix):
    # Image features input
    inputs1 = Input(shape=(49, 2048), name='image_input')
    
    # Add Dropout to image features → simulates data augmentation
    dropout_img = Dropout(0.5)(inputs1)
    
    attention = AttentionLayer()(dropout_img)

    # Sequence input
    inputs2 = Input(shape=(max_length,), name='seq_input')
    
    # EMBEDDING: Now TRAINABLE to correct GloVe bias!
    se1 = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder
    decoder1 = Concatenate()([attention, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Build model
model = define_model(vocab_size, max_length, embedding_matrix)
print(model.summary())

# Train model
epochs = 50  # Increased to 50 epochs!
steps = len(all_captions) // 64

generator = data_generator(captions, image_features, tokenizer, max_length, vocab_size, batch_size=64)

model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

# Save model
model.save('image_caption_model_with_attention.h5')
print('✅ Model saved as image_caption_model_with_attention.h5')