"""Flask web application for DBS prediction and LLaMA chatbot."""

import os
from flask import Flask, render_template, request
import joblib
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route("/main", methods=["GET", "POST"])
def main():
    """Render the main page."""
    return render_template("main.html")


@app.route("/llama", methods=["GET", "POST"])
def llama():
    """Render the LLaMA chat interface."""
    return render_template("llama.html")


@app.route("/llama_reply", methods=["GET", "POST"])
def llama_reply():
    """Process LLaMA chat request and return response."""
    q = request.form.get("q")
    # load model
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return render_template("llama_reply.html", r=completion.choices[0].message.content)


@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    """Render the DBS prediction interface."""
    return render_template("dbs.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """Process DBS prediction request and return result."""
    q = float(request.form.get("q"))

    # load model
    model = joblib.load("dbs.jl")

    # make prediction
    pred = model.predict([[q]])

    return render_template("prediction.html", r=pred)


if __name__ == "__main__":
    app.run()
