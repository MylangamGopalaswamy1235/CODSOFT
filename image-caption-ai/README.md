# Image Caption AI

A Flask app that uses OpenAI GPT-4o Vision to generate captions for uploaded images.

## Deploy to Vercel

1. Push this repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new) and import your GitHub repo
3. In **Environment Variables**, add:
   - Name: `OPENROUTER_API_KEY`
   - Value: `sk-or-your-real-key-here`
4. Click **Deploy**

That's it — no `.env` file needed on Vercel.

## Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your real OPENAI_API_KEY
python api/index.py
```

Then open: http://localhost:5000

## Features

- Drag & drop or browse to upload images (up to 20MB)
- Webcam capture support
- GPT-4o Vision generates a primary caption + 2 alternatives
- Copy caption to clipboard
- Read caption aloud with text-to-speech

## Supported Formats

PNG, JPG, JPEG, GIF, WEBP (max 20MB)
