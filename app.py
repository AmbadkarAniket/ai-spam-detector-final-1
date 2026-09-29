"""
app.py — Flask Web Server for the Spam Detector

This file creates a simple web server that:
1. Serves the HTML page to the user's browser.
2. Accepts messages via a POST /predict API endpoint.
3. Uses the trained ML model to classify messages as Spam or Not Spam.
4. Returns the prediction and confidence score as JSON.

How to run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
import joblib

# Import the shared ML classes and the text-cleaning function.
# These MUST be imported so that joblib can deserialize the
# saved model objects (they were instances of these classes).
from spam_model import SimpleTfidfVectorizer, SimpleNaiveBayes, clean_text

# ───────────────────────────────────────────────────────────────
# SETUP THE FLASK APP
# ───────────────────────────────────────────────────────────────

app = Flask(__name__)

# ───────────────────────────────────────────────────────────────
# LOAD THE TRAINED MODEL AND VECTORIZER
# ───────────────────────────────────────────────────────────────
# We load these once when the server starts so we don't have to
# read the files from disk every time someone sends a message.

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

model = None
vectorizer = None


def load_model():
    """
    Load the trained model and vectorizer from disk.
    Returns True if successful, False otherwise.
    """
    global model, vectorizer

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please run 'python train_model.py' first to train the model.")
        return False

    if not os.path.exists(VECTORIZER_PATH):
        print(f"ERROR: Vectorizer file not found at '{VECTORIZER_PATH}'")
        print("Please run 'python train_model.py' first to train the model.")
        return False

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


# ───────────────────────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint — accepts a message, returns spam/not-spam prediction.

    Expected JSON input:
        { "message": "Your message text here" }

    JSON output:
        { "prediction": "Spam", "confidence": 97.42 }
        or
        { "prediction": "Not Spam", "confidence": 94.15 }
    """

    # --- Check that the model is loaded ---
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model is not loaded. Please train the model first '
                     'by running: python train_model.py'
        }), 500

    # --- Parse the incoming JSON ---
    try:
        data = request.get_json()
    except Exception:
        return jsonify({'error': 'Invalid JSON in request body.'}), 400

    if data is None:
        return jsonify({
            'error': 'No JSON data received. Send a JSON body with a "message" field.'
        }), 400

    # --- Get the message from the request ---
    message = data.get('message', '')

    if not message or not message.strip():
        return jsonify({'error': 'Please enter a message to check.'}), 400

    # --- Clean the message (same way we cleaned training data) ---
    cleaned = clean_text(message)

    if not cleaned:
        return jsonify({
            'error': 'Message contains no usable text after cleaning.'
        }), 400

    # --- Make the prediction ---
    try:
        # Step 1: Convert the cleaned message to a TF-IDF vector
        message_tfidf = vectorizer.transform([cleaned])

        # Step 2: Get the predicted class ("spam" or "ham")
        prediction = model.predict(message_tfidf)[0]

        # Step 3: Get probability scores for each class
        probabilities = model.predict_proba(message_tfidf)[0]

        # The confidence is the probability of whichever class won
        confidence = round(max(probabilities.values()) * 100, 2)

        # Convert the internal label to user-friendly text
        result = 'Spam' if prediction == 'spam' else 'Not Spam'

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': 'An error occurred while making the prediction. '
                     'Please try again.'
        }), 500


# ───────────────────────────────────────────────────────────────
# START THE SERVER
# ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load the model when the server starts
    if not load_model():
        print("\nWARNING: The server will start, but predictions won't work.")
        print("  Run 'python train_model.py' to train the model first.\n")

    print("\nStarting the Spam Detector server...")
    print("Open http://127.0.0.1:5000 in your browser\n")

    # debug=True makes the server auto-reload when you edit code
    app.run(debug=True, host='127.0.0.1', port=5000)
