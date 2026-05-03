import base64
import io
import os

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

client = None


def get_openai_client():
    global client
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Set OPENROUTER_API_KEY in your Vercel environment variables.")
    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return client


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, GIF, or WEBP image."}), 400

    try:
        image_bytes = file.read()

        img = Image.open(io.BytesIO(image_bytes))
        img.verify()

        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "JPEG").lower()
        mime_map = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(fmt, "image/jpeg")
        image_data_url = image_to_base64(image_bytes, mime_type)

        openai_client = get_openai_client()

        # Single API call to get primary caption + 2 alternatives (avoids timeout)
        combined_response = openai_client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {
                            "type": "text",
                            "text": (
                                "Describe this image. Return ONLY a JSON object with no markdown, "
                                "no code fences, no extra text. Format:\n"
                                '{"caption": "one clear sentence", '
                                '"alternatives": ["alt sentence 1", "alt sentence 2"]}'
                            ),
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        raw = combined_response.choices[0].message.content.strip()

        import json
        # Strip markdown fences if model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        caption = result.get("caption", "")
        alternatives = result.get("alternatives", [])[:2]

        return jsonify({"caption": caption, "alternatives": alternatives})

    except Exception as e:
        error_msg = str(e)
        lowered = error_msg.lower()

        if "api key" in lowered or "api_key" in lowered or "authentication" in lowered or "401" in lowered:
            return jsonify({"error": "OpenRouter API key is invalid or missing. Set OPENROUTER_API_KEY in your Vercel environment variables."}), 500
        if "rate_limit" in lowered:
            return jsonify({"error": "OpenAI rate limit reached. Please try again in a moment."}), 429

        return jsonify({"error": f"Failed to generate caption: {error_msg}"}), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 20MB."}), 413


# For local development only
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
