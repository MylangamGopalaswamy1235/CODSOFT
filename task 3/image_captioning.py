# Load Flickr8k captions
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens
            image_id = image_id.split('#')[0]  # remove #0, #1, etc.
            caption = 'startseq ' + caption.strip() + ' endseq'
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions
# Load captions
captions_path = "Flicker8k_text/Flickr8k.token.txt"
captions = load_captions(captions_path)

print(f"Total images with captions: {len(captions)}")

# Example print
example_image = next(iter(captions))
print(f"\nExample image ID: {example_image}")
print("Captions:")
for cap in captions[example_image]:
    print(cap)