# Install necessary packages first if not done:
# pip install transformers gradio torch sentencepiece

from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define function to generate caption
def caption_image(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Create Gradio Interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="BLIP Image Captioning",
    description="Upload an image to get a caption using the BLIP model."
)

# Launch app
iface.launch(share=True)