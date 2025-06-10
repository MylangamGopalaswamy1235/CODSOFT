# create_tokenizer.py

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load captions
captions_mapping = {}
with open('Flicker8k_text/Flickr8k.token.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tokens = line.split('\t')
        image_id, caption = tokens[0].split('#')[0], tokens[1]
        caption = 'startseq ' + caption + ' endseq'
        captions_mapping.setdefault(image_id, []).append(caption)

# Prepare tokenizer
all_captions = []
for captions in captions_mapping.values():
    all_captions.extend(captions)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ tokenizer.pkl saved successfully.")