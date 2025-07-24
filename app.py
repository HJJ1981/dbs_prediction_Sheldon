"""Flask web application for DBS prediction and LLaMA chatbot."""

import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for
import joblib
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import requests
import numpy as np
from PIL import Image
import io
import base64

# Load environment variables from .env file (for local development)
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

SEALION_API_KEY = os.getenv('SEA_LION_API_KEY')
if not SEALION_API_KEY:
    raise ValueError("SEA_LION_API_KEY environment variable is not set")

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route("/main", methods=["GET", "POST"])
def main():
    """Render the main page."""
    name = request.form.get("q")
    if name:
        try:
            conn = sqlite3.connect("user.db")
            cursor = conn.cursor()
            # Insert the name with the current timestamp
            cursor.execute("INSERT INTO user (name, timestamp) VALUES (?, datetime('now'))", (name,))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
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


@app.route("/deepseek", methods=["GET", "POST"])
def deepseek():
    """Render the Deepseek chat interface."""
    return render_template("deepseek.html")


@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    """Process Deepseek chat request and return response."""
    q = request.form.get("q")
    # load model
    client = Groq()
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return render_template("deepseek_reply.html", r=completion.choices[0].message.content)


@app.route("/sealion", methods=["GET", "POST"])
def sealion():
    """Render the SEA-LION chat interface."""
    return render_template("sealion.html")


@app.route("/sealion_reply", methods=["GET", "POST"])
def sealion_reply():
    """Process SEA-LION chat request and return response."""
    q = request.form.get("q")
    # load model
    client = OpenAI(
        api_key=SEALION_API_KEY,
        base_url="https://api.sea-lion.ai/v1"
    )
    completion = client.chat.completions.create(
        model="aisingapore/Gemma-SEA-LION-v3-9B-IT",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return render_template("sealion_reply.html", r=completion.choices[0].message.content)


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


@app.route("/telegram", methods=["GET", "POST"])
def telegram():
    """Render the Telegram chat interface."""
    domain_url = 'https://dbs-prediction-cvhg.onrender.com'
    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    requests.post(delete_webhook_url, json={
                  "url": domain_url, "drop_pending_updates": True})
    # Set the webhook URL for the Telegram bot
    set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={domain_url}/webhook"
    webhook_response = requests.post(
        set_webhook_url, json={"url": domain_url, "drop_pending_updates": True})

    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot is running. Please check with the telegram bot. @SheldonGenAiBot"
    else:
        status = "Failed to start the telegram bot. Please check the logs."

    return render_template("telegram.html", status=status)


@app.route("/stop_telegram", methods=["GET", "POST"])
def stop_telegram():
    """Stop the Telegram bot by deleting the webhook."""
    domain_url = 'https://dbs-prediction-cvhg.onrender.com'

    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    webhook_response = requests.post(delete_webhook_url, json={
                                     "url": domain_url, "drop_pending_updates": True})

    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot is stopped. "
    else:
        status = "Failed to stop the telegram bot. Please check the logs."

    return render_template("telegram.html", status=status)


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    """This endpoint will be called by Telegram when a new message is received"""
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        # Extract the chat ID and message text from the update
        chat_id = update["message"]["chat"]["id"]
        query = update["message"]["text"]

        # Pass the query to the Groq model
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        response_message = completion_ds.choices[0].message.content

        # Send the response back to the Telegram chat
        send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(send_message_url, json={
            "chat_id": chat_id,
            "text": response_message
        })
    return ('ok', 200)


@app.route("/user_log", methods=["GET", "POST"])
def user_log():
    """Display user logs from the database."""
    users = []
    try:
        conn = sqlite3.connect("user.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        conn.close()
    except sqlite3.Error:
        users = []
    return render_template("user_log.html", users=users)


@app.route("/delete_log", methods=["GET", "POST"])
def delete_log():
    """Delete all user logs from the database."""
    try:
        conn = sqlite3.connect("user.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user")
        conn.commit()
        conn.close()
        status = "All user logs have been deleted."
    except sqlite3.Error as e:
        status = f"Error deleting logs: {e}"
    return render_template("delete_log.html", status=status)


def sepia_filter(input_img):
    """Apply sepia filter to an image."""
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_matrix.T)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img.astype(np.uint8)


@app.route("/sepia", methods=["GET", "POST"])
def sepia():
    """Render the sepia filter interface."""
    return render_template("sepia.html")


@app.route("/sepia_result", methods=["POST"])
def sepia_result():
    """Process uploaded image and apply sepia filter."""
    if 'image' not in request.files:
        return redirect(url_for('sepia'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('sepia'))
    
    try:
        # Read and process the image
        image = Image.open(file.stream)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply sepia filter
        sepia_img_array = sepia_filter(img_array)
        
        # Convert back to PIL Image
        sepia_image = Image.fromarray(sepia_img_array)
        
        # Convert images to base64 for display
        def img_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        original_b64 = img_to_base64(image)
        sepia_b64 = img_to_base64(sepia_image)
        
        return render_template("sepia_result.html", 
                             original_image=original_b64, 
                             sepia_image=sepia_b64)
    
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        return render_template("sepia.html", error=error_message)


if __name__ == "__main__":
    app.run()

# Set webhook for Telegram bot
# https://api.telegram.org/bot{groq_telegram_token}/setWebhook?url ={domain_url}/webhook

# Delete webhook for Telegram bot
# https: //api.telegram.org/bot%7Bgroq_telegram_token%7D/deleteWebhook
