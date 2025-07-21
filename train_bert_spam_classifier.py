import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t',
                 header=None, names=['label', 'text'])

# Encode labels
df['label'] = df['label'].map({'ham': 'ham', 'spam': 'spam'})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Load BERT encoder
encoder = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode messages
X_train_emb = encoder.encode(X_train.tolist())
X_test_emb = encoder.encode(X_test.tolist())

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)

# Evaluate
y_pred = clf.predict(X_test_emb)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'model_bert_lr.pkl')
print("Model saved as model_bert_lr.pkl")
