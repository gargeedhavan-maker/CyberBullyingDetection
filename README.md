# Multilingual Cyber Bullying Detection System Using NLP

A complete, beginner-friendly Natural Language Processing (NLP) college project for classifying social media comments into **Cyber Bullying** or **Not Cyber Bullying** across multiple Indian languages:
1. **English**
2. **Hindi** (Devanagari script & Romanized / Hinglish)
3. **Marathi** (Devanagari script & Romanized / Marathlish)

Built using **Python**, **Flask**, **TF-IDF Vectorizer**, and **Logistic Regression**, enhanced with an Instagram-style social media feed, pre-posting 3-tier comment moderation, and automated multilingual testing.

---

## 1. Project Objective & Workflow

The system takes social media comments as input in English, Hindi, or Marathi (in Devanagari or Romanized script) and processes them through an NLP text classification pipeline:

```text
Social Media Comment (English / Hindi / Marathi)
        ↓
Script-Aware Language Estimation
        ↓
Unicode-Preserving Preprocessing (Keeps Devanagari \u0900-\u097F)
        ↓
Multilingual TF-IDF N-Gram Vectorization
        ↓
Logistic Regression Classifier + Moderation Rules
        ↓
3-Tier Moderation Outcome:
  - SAFE    : Comment appears safe → confirmed & posted to feed
  - REVIEW  : Potentially derogatory critique → review/rephrase prompt
  - BLOCKED : Clear abuse, threat, or harassment → blocked from posting
```

---

## 2. Key Features

### Feature 1: Multilingual Comment Moderation
- Moderates comments in:
  - **English**: *"Beautiful picture!"*, *"You are useless and nobody likes you."*, *"I disagree with your opinion."*
  - **Hindi (Devanagari)**: *"तुम्हारी फोटो बहुत अच्छी है।"*, *"तुम किसी काम के नहीं हो।"*, *"मुझे आपकी बात से सहमत नहीं हूँ।"*
  - **Marathi (Devanagari)**: *"तुझा फोटो खूप छान आहे."*, *"तू काहीच कामाचा नाहीस."*, *"मला तुझं मत मान्य नाही."*
  - **Romanized Hindi**: *"Tum bahut achhe ho."*, *"Tum kisi kaam ke nahi ho."*, *"Mujhe tumhari baat se sehmat nahi hai."*
  - **Romanized Marathi**: *"Tujha photo khup chhan aahe."*, *"Tu kahich kaamacha nahis."*, *"Mala tujh mat manya nahi."*
- **3-Tier Outcomes**:
  - `SAFE`: Approved for publication after confirmation.
  - `REVIEW`: Prompts user to review and rephrase (does not issue strikes for uncertain predictions).
  - `BLOCKED`: Blocks clear personal attacks, abusive language, or threats.
- Displays original comment text in the feed exactly as typed without translation.

### Feature 2: Custom Multilingual Dataset (`multilingual_cyberbullying_dataset.csv`)
- Contains 534 carefully reviewed educational samples with 4 columns:
  - `text`: Original social media comment.
  - `language`: `English`, `Hindi`, `Marathi`, `Romanized Hindi`, `Romanized Marathi`.
  - `label`: `safe`, `bullying`.
  - `category`: `normal`, `advice`, `disagreement`, `criticism`, `insult`, `harassment`, `threat`.
- Balanced distribution between safe and bullying across all writing styles.
- Custom educational dataset for academic study and moderation testing.

### Feature 3: Pre-Populated Feed Comments
- Displays 3–5 pre-approved sample comments under appropriate feed posts.
- Natural mix of English, Hindi, and Marathi in original scripts (e.g., *"तुझा फोटो खूप छान आहे."*, *"आजचा दिवस खूप छान होता."*, *"तुम्हारी फोटो बहुत अच्छी है।"*, *"Beautiful picture!"*).
- Separates pre-approved comments from newly submitted comments.

### Feature 4: Automated Moderation Test Suite (`test_multilingual_moderation.py`)
- Automated test script validating 16 distinct test cases across all supported languages, verifying response formats, language detection, 3-tier states, and policy safeguards.

---

## 3. Project Structure

```text
CyberBullyingDetection/
│
├── app.py                                   # Flask web server, multilingual moderation routes & feed APIs
├── train_model.py                           # Unicode-preserving preprocessing, TF-IDF vectorization, & model training
├── test_multilingual_moderation.py         # Automated test script for all 5 language variants
├── multilingual_cyberbullying_dataset.csv  # Custom 534-row educational dataset
├── requirements.txt                         # Lightweight Python dependencies
├── README.md                                # Setup guide & documentation
│
├── bullying_model.pkl                       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                     # Multilingual TF-IDF vectorizer
├── metrics.json                             # Evaluation scores & language breakdown
│
├── templates/
│   ├── index.html                           # Home page single detector
│   ├── feed.html                            # Instagram-style feed with multilingual moderation modal
│   ├── how_it_works.html                    # 4-step NLP pipeline explanation
│   ├── history.html                         # Detection history table
│   ├── awareness.html                       # Cyber safety awareness
│   ├── examples.html                        # Try sample comments in multiple languages
│   └── performance.html                     # Model accuracy & evaluation report
│
└── static/
    └── style.css                            # Clean dark-mode stylesheet
```

---

## 4. Visual Theme & Palette

Strict minimalist dark aesthetic:
- **Background**: `#111111`
- **Cards & Posts**: `#1C1C1C`
- **Inputs**: `#252525`
- **Borders**: `#3A3A3A`
- **Primary Text**: `#FFFFFF`
- **Secondary Text**: `#AAAAAA`
- **Buttons**: `#333333` (Hover: `#444444`)

---

## 5. How to Run Locally in VS Code on Windows

### Step 1: Open Project in VS Code
1. Open VS Code.
2. Select **File > Open Folder...** and choose the `CyberBullyingDetection` folder.
3. Open a new terminal with ``Ctrl + ` ``.

### Step 2: Create & Activate Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```
*(If PowerShell shows execution policy restrictions, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `venv\Scripts\activate` again).*

### Step 3: Install Required Packages
```cmd
pip install -r requirements.txt
```

### Step 4: Train the Multilingual NLP Model
```cmd
python train_model.py
```
This trains the model on `multilingual_cyberbullying_dataset.csv`, computes accuracy, precision, recall, and F1 score on unseen holdout test data, and saves `bullying_model.pkl` and `tfidf_vectorizer.pkl`.

### Step 5: Run Automated Tests
```cmd
python test_multilingual_moderation.py
```
Verifies 16 test cases across English, Hindi, Marathi, and Romanized comments with 100% compliance.

### Step 6: Start the Flask Application
```cmd
python app.py
```

### Step 7: Open in Browser
Open your browser and navigate to:
- **Home Detector**: `http://127.0.0.1:5000/`
- **Social Media Feed**: `http://127.0.0.1:5000/feed`

---

## 6. System Limitations & Academic Notes

For your college viva and project presentation, note these real-world NLP limitations:
- **Spelling Variations in Romanized Text**: Informal Romanized Hindi and Marathi have no standardized spelling (e.g., *"achha"*, *"acha"*, *"accha"*). Character n-grams help mitigate this, but rare phonetic spellings can reduce model confidence.
- **Code-Switching**: Sentences mixing English, Hindi, and Marathi words within the same clause require rich multilingual context.
- **Sarcasm and Context**: Pure bag-of-words / TF-IDF representations analyze word co-occurrences and n-grams without capturing sarcastic tonality.
- **Regional Dialects**: Dialects across Maharashtra (e.g., Varhadi, Malvani) and northern India have colloquial terms beyond standard educational dictionaries.
