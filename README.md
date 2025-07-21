# DBS Prediction & AI Chat Platform

A Flask web application that combines DBS (Deep Brain Stimulation) prediction, SMS spam detection, and multiple AI chatbot interfaces including LLaMA, DeepSeek, and SEA-LION models.

## Features

- **DBS Prediction**: Machine learning model to predict DBS-related outcomes
- **SMS Spam Detection**: BERT-based spam classifier for text messages
- **AI Chatbots**: Integration with multiple LLMs:
  - LLaMA 3.1 8B (via Groq)
  - DeepSeek R1 Distill LLaMA 70B (via Groq)
  - SEA-LION v3 9B (Singapore AI model)
- **Telegram Bot**: Automated bot integration
- **Web Interface**: Clean, responsive HTML templates

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- API keys for:
  - Groq (for LLaMA and DeepSeek models)
  - SEA-LION API
  - Telegram Bot (optional)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd dbs_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
SEA_LION_API_KEY=your_sealion_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 4. Download Spam Dataset (First Time Only)
The SMS spam classifier requires the SMSSpamCollection dataset:
```bash
# Download and extract dataset
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
```

### 5. Train Spam Classifier (First Time Only)
```bash
python train_bert_spam_classifier.py
```
This will:
- Load the SMS spam dataset
- Train a BERT-based classifier
- Save the model as `model_bert_lr.pkl`

### 6. Run the Application
```bash
python app.py
```

Open your browser and go to `http://localhost:5000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/main` | GET | Main dashboard |
| `/dbs` | GET/POST | DBS prediction interface |
| `/prediction` | POST | Process DBS prediction |
| `/spam` | GET/POST | Spam detection interface |
| `/spam_predict` | POST | Predict if message is spam |
| `/llama` | GET/POST | LLaMA chat interface |
| `/llama_reply` | POST | Get LLaMA response |
| `/deepseek` | GET/POST | DeepSeek chat interface |
| `/deepseek_reply` | POST | Get DeepSeek response |
| `/sealion` | GET/POST | SEA-LION chat interface |
| `/sealion_reply` | POST | Get SEA-LION response |
| `/telegram` | GET/POST | Start Telegram bot |
| `/stop_telegram` | GET/POST | Stop Telegram bot |

## Project Structure

```
dbs_prediction/
├── app.py                          # Main Flask application
├── train_bert_spam_classifier.py   # BERT spam classifier training
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .env                           # Your API keys (create from .env.example)
├── SMSSpamCollection              # SMS spam dataset (downloaded)
├── model_bert_lr.pkl              # Trained BERT spam classifier
├── dbs.jl                         # DBS prediction model
├── cv_encoder.pkl                 # CV encoder for DBS model
├── lr_model.pkl                   # Logistic regression model
├── templates/                     # HTML templates
│   ├── index.html
│   ├── main.html
│   ├── dbs.html
│   ├── spam.html
│   ├── llama.html
│   ├── deepseek.html
│   └── sealion.html
└── static/                        # CSS, JS, images
```

## Usage Examples

### DBS Prediction
1. Navigate to `/dbs`
2. Enter a numerical value
3. Get prediction result

### Spam Detection
1. Navigate to `/spam`
2. Enter any text message
3. Get spam/ham classification with BERT model

### AI Chat
1. Choose your preferred model:
   - `/llama` for LLaMA 3.1 8B
   - `/deepseek` for DeepSeek R1
   - `/sealion` for SEA-LION v3 9B
2. Enter your question
3. Get AI-generated response

## Troubleshooting

### Common Issues

**"SMSSpamCollection not found"**
- Download the dataset as shown in step 4 above

**"BERT model not found"**
- Run the training script: `python train_bert_spam_classifier.py`

**"API key not set"**
- Check your `.env` file has the correct API keys
- Ensure `.env` file is in the project root

**"Module not found"**
- Install dependencies: `pip install -r requirements.txt`

### Model Performance

The BERT spam classifier achieves:
- **99% overall accuracy**
- **99% precision/recall** for legitimate messages
- **97% precision, 94% recall** for spam messages

## Development

### Adding New Models
1. Add route in `app.py`
2. Create corresponding HTML template
3. Update requirements.txt if needed

### Deployment
The app is configured for deployment on platforms like Render, Heroku, or similar.

## Dependencies

- **Flask**: Web framework
- **sentence-transformers**: BERT embeddings
- **scikit-learn**: Machine learning models
- **groq**: LLaMA and DeepSeek API client
- **openai**: SEA-LION API client
- **joblib**: Model serialization
- **python-dotenv**: Environment variable management
- **requests**: HTTP requests
- **gunicorn**: WSGI server for deployment

## License

This project is for educational and research purposes.
