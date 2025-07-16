"""Flask web application for DBS prediction and LLaMA chatbot."""

import os
from flask import Flask, render_template, request
import joblib
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import requests

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
    """Stop the Telegram chatbot."""
    domain_url = 'https://dbs-prediction-cvhg.onrender.com'
    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    webhook_response = requests.post(delete_webhook_url, json={
                  "url": domain_url, "drop_pending_updates": True})

    if webhook_response == 200:
        # set status message
        status = "The telegram bot is now stopped."
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


if __name__ == "__main__":
    app.run()

# Set webhook for Telegram bot
# https://api.telegram.org/bot{groq_telegram_token}/setWebhook?url ={domain_url}/webhook

# Delete webhook for Telegram bot
# https: //api.telegram.org/bot%7Bgroq_telegram_token%7D/deleteWebhook
