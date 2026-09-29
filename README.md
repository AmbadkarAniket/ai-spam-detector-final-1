# 🛡️ AI Spam Message Detector

A beginner-friendly web application that uses **Machine Learning** to detect whether a message is **Spam** or **Not Spam**, along with a confidence percentage.

Built with **Python (Flask + scikit-learn)** for the backend and **HTML, CSS, JavaScript** for the frontend.

---

## ✨ Features

- 🤖 **AI-Powered Detection** — Uses a trained Naive Bayes model to classify messages.
- 📊 **Confidence Score** — Shows how certain the model is about its prediction.
- 🎨 **Modern Dark UI** — Clean, responsive design with smooth animations.
- ⚡ **Real-Time Predictions** — Get results instantly via the web interface.
- 🧹 **Text Preprocessing** — Automatic text cleaning for better accuracy.
- 📱 **Mobile Friendly** — Works well on all screen sizes.

---

## 🛠️ Technologies Used

| Layer    | Technology                                           |
| -------- | ---------------------------------------------------- |
| Backend  | Python, Flask                                        |
| ML Model | Naive Bayes (implemented from scratch)               |
| NLP      | TF-IDF Vectorizer (implemented from scratch)         |
| Frontend | HTML5, CSS3, Vanilla JavaScript                      |
| Data     | Python standard library (csv, math, collections)     |
| Storage  | joblib (model serialization)                         |

---

## 📁 Folder Structure

```
ai spam dectetor/
│
├── app.py                  ← Flask web server
├── train_model.py          ← Model training script
├── spam_model.py           ← Shared ML classes (TF-IDF + Naive Bayes)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── dataset/
│   └── spam.csv            ← Training dataset (spam/ham messages)
│
├── model/
│   ├── spam_model.pkl      ← Trained ML model (created after training)
│   └── vectorizer.pkl      ← TF-IDF vectorizer (created after training)
│
├── templates/
│   └── index.html          ← Main web page
│
└── static/
    ├── style.css           ← Stylesheet
    └── script.js           ← Frontend logic
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+** installed on your system
- **pip** (comes with Python)

### Step 1 — Clone or Download the Project

Place the project folder on your machine.

### Step 2 — Create a Virtual Environment

Open a terminal in the project folder and run:

```bash
python -m venv venv
```

### Step 3 — Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Train the Model

```bash
python train_model.py
```

You should see output like:

```
Loading dataset...
  → Loaded 120 messages
  → Spam: 30
  → Ham (not spam): 90

Cleaning text...
Splitting data into training and testing sets...
  → Training samples: 96
  → Testing samples:  24

Converting text to numerical features (TF-IDF)...
Training the Naive Bayes model...
Model training completed!

Accuracy: 0.96

Classification Report:
...

Model saved to:      model/spam_model.pkl
Vectorizer saved to: model/vectorizer.pkl
```

### Step 6 — Run the Application

```bash
python app.py
```

### Step 7 — Open in Browser

Visit: **http://127.0.0.1:5000**

---

## 🧠 How the AI Model Works

```
User Message
      ↓
HTML + JavaScript (frontend)
      ↓
Flask API (/predict endpoint)
      ↓
Text Cleaning (lowercase, remove symbols)
      ↓
TF-IDF Vectorizer (converts text → numbers)
      ↓
Naive Bayes Model (classifies the message)
      ↓
Spam / Not Spam + Confidence Score
      ↓
JSON Response → Web UI
```

### Key Concepts

1. **TF-IDF (Term Frequency–Inverse Document Frequency)**
   Converts text into numbers. Words that appear frequently in a message but rarely across all messages get higher scores — making them more useful for classification.

2. **Multinomial Naive Bayes**
   A simple probability-based algorithm. It calculates: *"Given these words, what's the probability this message is spam?"* It's fast, effective, and perfect for text classification.

3. **Text Cleaning**
   Before the model sees a message, we clean it: convert to lowercase, remove numbers and symbols, and collapse whitespace. This reduces noise and improves accuracy.

---

## 📝 Example Messages

### Spam Messages (try these!)

```
Congratulations! You won a free iPhone! Click here to claim your prize!
```

```
URGENT: Your bank account has been compromised. Call 1-800-555-0199 immediately!
```

```
You have been selected to receive a £900 prize reward! Call 09061701461 now!
```

```
FREE entry to win cash prizes! Text WIN to 85233. Cost 150p/day.
```

### Normal Messages (not spam)

```
Hey, are we still meeting for lunch tomorrow?
```

```
Can you send me the notes from today's lecture?
```

```
I'll be late by about 10 minutes. Traffic is terrible today.
```

```
Happy birthday! Hope you have an amazing day!
```

---

## ⚠️ Troubleshooting

| Problem                              | Solution                                              |
| ------------------------------------ | ----------------------------------------------------- |
| `ModuleNotFoundError`                | Run `pip install -r requirements.txt`                 |
| Model files not found                | Run `python train_model.py` first                     |
| Port 5000 already in use             | Change the port in `app.py` or close the other app    |
| Low accuracy                         | Add more training data to `dataset/spam.csv`          |

---

## 📜 License

This project is for educational purposes. Feel free to use and modify it for your college projects!
