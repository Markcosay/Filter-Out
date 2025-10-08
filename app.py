from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import requests
import pickle
import re

# -----------------------------------
# Flask app setup
# -----------------------------------
app = Flask(__name__)

# -----------------------------------
# Load ML Model and Tokenizer
# -----------------------------------
MODEL_PATH = "toxicity.h5"          # your model file
TOKENIZER_PATH = "tokenizer.pkl"    # your tokenizer file

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # must match training setup

# -----------------------------------
# YouTube API Setup
# -----------------------------------
# https://console.cloud.google.com/apis/credentials
API_KEY = "Needed" 
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# -----------------------------------
# Helper: YouTube comment fetcher
# -----------------------------------
def scrape_comments_youtube(url):
    """Fetch top-level comments from a YouTube video."""
    try:
        match = re.search(r"(?:v=|youtu\.be/)([^&]+)", url)
        if not match:
            return []

        video_id = match.group(1)
        comments = []

        request = YOUTUBE.commentThreads().list(
            part="snippet", videoId=video_id, textFormat="plainText", maxResults=100
        )
        response = request.execute()

        for item in response["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)

        return comments

    except Exception as e:
        print("YouTube API error:", e)
        return []

# -----------------------------------
# Helper: Generic HTML comment scraper
# -----------------------------------
def scrape_comments_generic(url):
    """Try to scrape comments or text from generic web pages."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        comments = []

        for tag in soup.find_all(["p", "div", "span"]):
            if tag.get("class") and any("comment" in c.lower() for c in tag.get("class")):
                comments.append(tag.get_text(strip=True))
            elif tag.get("id") and "comment" in tag.get("id").lower():
                comments.append(tag.get_text(strip=True))

        # fallback: grab paragraphs
        if not comments:
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            comments = [t for t in paragraphs if len(t.split()) > 3]

        return comments

    except Exception as e:
        print("Scraping error:", e)
        return []

# -----------------------------------
# Helper: Run toxicity detection
# -----------------------------------
def predict_toxicity(comments):
    toxic_comments = []
    for text in comments:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        score = float(model.predict(padded)[0][0])
        if score >= 0.5:
            toxic_comments.append({"text": text, "score": round(score, 2)})
    return toxic_comments

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")

        # Step 1: Identify platform
        if "youtube.com" in url or "youtu.be" in url:
            comments = scrape_comments_youtube(url)
        else:
            comments = scrape_comments_generic(url)

        # Step 2: Run toxicity detection
        toxic_comments = predict_toxicity(comments)

        return render_template(
            "results.html",
            url=url,
            toxic_comments=toxic_comments,
            total=len(comments),
        )

    return render_template("index.html")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
