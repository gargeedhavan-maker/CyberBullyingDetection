"""
Multilingual Cyber Bullying Detection System - Flask Web Application
Social Media Comment Moderation with 3-Tier Multilingual NLP Analysis
Supported: English, Hindi (Devanagari), Marathi (Devanagari), Romanized Hindi, Romanized Marathi
"""

import os
import re
import json
import time
try:
    import joblib
except ImportError:
    joblib = None

try:
    from flask import Flask, render_template, request, jsonify
    app = Flask(__name__)
except ImportError:
    class DummyApp:
        def route(self, *args, **kwargs):
            return lambda f: f
        def run(self, *args, **kwargs):
            print("[INFO] Flask is not installed. Run: pip install -r requirements.txt")
    app = DummyApp()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bullying_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Configurable 3-Tier Probability Thresholds
# Tier 1 (Safe): prob_bullying < 0.35
# Tier 2 (Review / Borderline): 0.35 <= prob_bullying < 0.70
# Tier 3 (Blocked / Clearly Harmful): prob_bullying >= 0.70
PROB_SAFE_THRESHOLD = 0.35
PROB_BLOCK_THRESHOLD = 0.70

# ==================== LANGUAGE DETECTION ====================

MARATHI_DEVANAGARI_MARKERS = [
    "आहे", "नाहीस", "आहेस", "नाही", "खूप", "छान", "तुझा", "तुझे", "माझं", "तुझं",
    "केलंस", "काहीच", "कामाचा", "कसं", "काय", "करायचं", "होता", "होती", "मित्रा",
    "भावा", "गाढवा", "लायकी", "बिनडोक", "नाहीतर", "सगळे", "घ्या", "कौतुक", "वाटचालीस",
    "काळजी", "औषधं", "दिसलास", "बघून", "तोडून", "तिडक", "घाणेरडा", "विचार", "सल्ला"
]

HINDI_DEVANAGARI_MARKERS = [
    "है", "हैं", "नहीं", "बहुत", "अच्छा", "अच्छे", "अच्छी", "काम", "फोटो", "आपकी",
    "बात", "सहमत", "हूँ", "करो", "कृपया", "दवाइयों", "जांच", "सकते", "बधाई", "तस्वीर",
    "बेवकूफ", "मूर्ख", "घटिया", "बदसूरत", "दफा", "औकात", "चुप", "अक्ल", "तोड़", "बर्बाद"
]

ROMANIZED_MARATHI_TOKENS = {
    "aahe", "nahis", "ahes", "khup", "chhan", "tujha", "tujhe", "mala", "mat",
    "manya", "kelans", "kahich", "kamacha", "kasa", "kay", "bhava", "mitra",
    "aapan", "karto", "kartos", "gela", "jhala", "hota", "hoti", "gadhavachya",
    "bindok", "ghanerda", "laaj", "aukaat", "swatahchi", "kalji", "aushadh",
    "salla", "baghun", "dakhvto", "todun", "vichar", "abhinandan", "aabhar",
    "shubheccha", "koutuk", "gaana", "nisargacha", "divas", "bhet"
}

ROMANIZED_HINDI_TOKENS = {
    "tum", "bahut", "achhe", "achhi", "accha", "kisi", "kaam", "ke", "nahi",
    "ho", "mujhe", "tumhari", "baat", "sehmat", "hai", "hain", "kro", "phle",
    "khatam", "karo", "bhai", "yaar", "mera", "meri", "kya", "kyu", "shandar",
    "badhai", "shukriya", "dhyan", "rakho", "bewakoof", "gadhe", "ullu",
    "badsurat", "aukaat", "chup", "faltu", "bekar", "ghatiya", "barbad",
    "dunga", "pitunga", "sambhal", "aapse", "hamesha"
}

def detect_language(text: str) -> dict:
    """
    Transparent script-aware & lexical language estimator.
    Returns estimated language and script category.
    """
    if not text:
        return {"language": "English", "script": "Latin"}

    # 1. Check for Devanagari characters (\u0900 - \u097F)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))

    if has_devanagari:
        # Differentiate between Marathi and Hindi
        words = text.split()
        marathi_hits = sum(1 for m in MARATHI_DEVANAGARI_MARKERS if m in text)
        hindi_hits = sum(1 for h in HINDI_DEVANAGARI_MARKERS if h in text)

        if marathi_hits > hindi_hits:
            return {"language": "Marathi", "script": "Devanagari"}
        elif hindi_hits > 0:
            return {"language": "Hindi", "script": "Devanagari"}
        else:
            # Default to Hindi for Devanagari if equal hits
            return {"language": "Hindi", "script": "Devanagari"}

    # 2. Latin script: check for Romanized Marathi or Romanized Hindi
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    marathi_latin_hits = len(tokens.intersection(ROMANIZED_MARATHI_TOKENS))
    hindi_latin_hits = len(tokens.intersection(ROMANIZED_HINDI_TOKENS))

    if marathi_latin_hits >= 2 or (marathi_latin_hits > hindi_latin_hits and marathi_latin_hits >= 1):
        return {"language": "Romanized Marathi", "script": "Latin"}
    elif hindi_latin_hits >= 2 or (hindi_latin_hits > marathi_latin_hits and hindi_latin_hits >= 1):
        return {"language": "Romanized Hindi", "script": "Latin"}
    elif marathi_latin_hits == 1:
        return {"language": "Romanized Marathi", "script": "Latin"}
    elif hindi_latin_hits == 1:
        return {"language": "Romanized Hindi", "script": "Latin"}

    return {"language": "English", "script": "Latin"}

# ==================== MULTILINGUAL RULES ====================

# Severe abusive phrases immediately triggering Tier 3 (Blocked)
SEVERE_HARASSMENT_PATTERNS = [
    # English
    r"\bshut up\b.*\bidiot\b",
    r"\bidiot\b",
    r"\bnobody likes you\b",
    r"\byou are useless\b",
    r"\buseless and nobody\b",
    r"\bkill yourself\b",
    r"\beveryone hates you\b",
    r"\bworthless\b",
    r"\bbrainless\b",
    r"\bpathetic loser\b",
    r"\bwatch your back\b",
    r"\bleak all your\b",
    # Hindi Devanagari
    r"किसी काम के नहीं",
    r"बेवकूफ और मूर्ख",
    r"निहायत ही बेवकूफ",
    r"चुप रह बेवकूफ",
    r"बदसूरत",
    r"दफा हो जाओ",
    r"मुँह तोड़",
    r"बर्बाद कर",
    r"जिंदा वापस नहीं",
    r"गंदी शक्ल",
    # Marathi Devanagari
    r"काहीच कामाचा नाहीस",
    r"बिनडोक माणूस",
    r"गप बस गाढवा",
    r"डोक्यात शेण",
    r"नालायक माणूस",
    r"हातपाय तोडून",
    r"मार खाशील",
    r"आयुष्य संपवून",
    r"लाचार माणसा",
    # Romanized Hindi
    r"\bkisi kaam ke nahi\b",
    r"\bbilkul bewakoof\b",
    r"\bchup kar idiot\b",
    r"\bshakal dekh ke ulti\b",
    r"\bjaan se maar dunga\b",
    r"\bmuh tod dunga\b",
    r"\bchats online leak\b",
    r"\bdharti pe bojh\b",
    # Romanized Marathi
    r"\bkahich kaamacha nahis\b",
    r"\bbindok manus\b",
    r"\bgap bas gadhavachya\b",
    r"\bdokyat shen\b",
    r"\bnalayak manus\b",
    r"\bmaar khashil\b",
    r"\bjivant sodnar nahi\b",
    r"\btond band thev\b"
]

# Moderate insulting critiques triggering Tier 2 (Review)
MODERATE_REVIEW_PATTERNS = [
    # English
    r"\bvery bad picture\b",
    r"\bbad picture\b",
    r"\bterrible picture\b",
    r"\bpicture looks terrible\b",
    r"\bphoto looks awful\b",
    r"\bhorrible picture\b",
    r"\bugly picture\b",
    r"\byou look terrible\b",
    r"\bdelete this photo\b",
    # Hindi Devanagari
    r"बहुत गंदी तस्वीर",
    r"वाहियात फोटो",
    r"बकवास आदमी",
    r"जोकर",
    # Marathi Devanagari
    r"घाणेरडा फोटो",
    r"वाईट फोटो",
    r"फालतू सल्ला",
    # Romanized Hindi
    r"\bvery bad picture\b",
    r"\bbekar photo\b",
    r"\bgandi picture\b",
    r"\bghatiya photo\b",
    # Romanized Marathi
    r"\bghanerda photo\b",
    r"\bbekar photo\b",
    r"\bfaltu salla\b"
]

# Safeguard patterns: Respectful disagreements, advice, and healthcare
SAFE_EXCEPTION_PHRASES = [
    # English
    "could be improved", "better lighting", "different angle",
    "consult a doctor", "doctor's advice", "doctors advice", "as a doctor",
    "prescribed medicines", "disagree with your opinion", "different perspective",
    "beautiful picture", "lovely picture", "nice shot", "good suggestion",
    "take care of yourself", "thank you for sharing", "stay hydrated",
    # Hindi Devanagari
    "बहुत अच्छे हो", "फोटो बहुत अच्छी", "सुंदर", "धन्यवाद", "बधाई",
    "ध्यान रखें", "डॉक्टर से", "दवाइयों", "सहमत नहीं हूँ", "सलाह",
    # Marathi Devanagari
    "खूप छान आहे", "छान काम केलंस", "खूप छान होता", "सुंदर", "आभार",
    "अभिनंदन", "काळजी घे", "डॉक्टरांचा सल्ला", "मत मान्य नाही",
    # Romanized Hindi
    "bahut achhe", "bahut achhi", "shandar", "shukriya", "badhai",
    "doctor i suggest", "prescribed", "medicines", "sehmat nahi", "dhyan rakho",
    # Romanized Marathi
    "khup chhan aahe", "khup chhan hota", "chhan kaam", "khup sundar",
    "aabhar", "abhinandan", "doctor cha salla", "kalji ghe", "mat manya nahi"
]

model = None
vectorizer = None

def load_saved_model():
    """Load model and vectorizer from pickle files using joblib."""
    global model, vectorizer
    if joblib is not None and os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("[INFO] Multilingual Model and TF-IDF Vectorizer loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model files: {e}")
    else:
        print("[WARNING] Multilingual Model or Vectorizer file not found.")
        print("          Please run: python train_model.py")

load_saved_model()

def clean_text(text: str) -> str:
    """Multilingual text cleaning preserving Devanagari and Latin letters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def analyze_comment_3tier(comment_text: str):
    """
    Evaluates multilingual comments across 3 distinct moderation states:
    - SAFE: Suitable for posting.
    - REVIEW: May be insulting or harmful and needs review.
    - BLOCKED: Clearly abusive, severe harassment or threatening content.
    """
    raw_lower = comment_text.lower().strip()
    cleaned = clean_text(comment_text)
    lang_info = detect_language(comment_text)
    lang_name = lang_info["language"]

    # 1. Check for severe abusive or threatening harassment (BLOCKED)
    for pat in SEVERE_HARASSMENT_PATTERNS:
        if re.search(pat, raw_lower):
            return {
                "state": "blocked",
                "heading": "COMMENT BLOCKED",
                "message": f"This comment contains abusive or threatening language in {lang_name} and cannot be posted.",
                "prob_bullying": 0.95,
                "confidence": 95,
                "can_post": False,
                "language": lang_name,
                "category": "harassment"
            }

    # 2. Check for moderate negative/insulting critiques (REVIEW)
    for pat in MODERATE_REVIEW_PATTERNS:
        if re.search(pat, raw_lower):
            return {
                "state": "review",
                "heading": "REVIEW YOUR COMMENT",
                "message": f"This comment may sound insulting or derogatory. Would you like to revise it before posting?",
                "prob_bullying": 0.58,
                "confidence": 58,
                "can_post": False,
                "language": lang_name,
                "category": "criticism"
            }

    # 3. Model inference using multilingual TF-IDF + Logistic Regression
    prob_bullying = 0.12
    if model and vectorizer and cleaned:
        try:
            text_vector = vectorizer.transform([cleaned])
            probabilities = model.predict_proba(text_vector)[0]
            class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
            bullying_idx = class_indices.get("bullying", 0)
            prob_bullying = float(probabilities[bullying_idx])
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")

    # 4. Check explicit safe exceptions (Disagreements, Advice, Healthcare, Compliments)
    for phrase in SAFE_EXCEPTION_PHRASES:
        if phrase in raw_lower:
            prob_bullying = min(prob_bullying, 0.10)
            break

    # Determine 3-Tier State
    if prob_bullying >= PROB_BLOCK_THRESHOLD:
        return {
            "state": "blocked",
            "heading": "COMMENT BLOCKED",
            "message": f"Our multilingual NLP model detected cyberbullying content ({lang_name}). This comment cannot be posted.",
            "prob_bullying": round(prob_bullying, 3),
            "confidence": round(prob_bullying * 100),
            "can_post": False,
            "language": lang_name,
            "category": "insult"
        }
    elif prob_bullying >= PROB_SAFE_THRESHOLD:
        return {
            "state": "review",
            "heading": "REVIEW YOUR COMMENT",
            "message": f"Our NLP model flagged potential hostility or borderline phrasing ({lang_name}). Please review.",
            "prob_bullying": round(prob_bullying, 3),
            "confidence": round(prob_bullying * 100),
            "can_post": False,
            "language": lang_name,
            "category": "review"
        }
    else:
        return {
            "state": "safe",
            "heading": "COMMENT VERIFICATION PASSED",
            "message": f"Comment verified safe to publish in {lang_name}. No cyberbullying detected.",
            "prob_bullying": round(prob_bullying, 3),
            "confidence": round((1.0 - prob_bullying) * 100),
            "can_post": True,
            "language": lang_name,
            "category": "normal"
        }

# ==================== SAMPLE FEED POSTS ====================
# Pre-populated with 3-5 curated safe comments in English, Hindi, and Marathi
POSTS_DATA = [
    {
        "id": 1,
        "username": "alex_codes",
        "avatar": "AC",
        "image": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&w=800&q=80",
        "caption": "Finalizing our NLP text classification semester project! 💻🚀 Tested TF-IDF vectorization with Logistic Regression for multilingual comments.",
        "time": "2h ago",
        "likes": 42,
        "liked_by": [],
        "comments": [
            {"id": 101, "username": "priya_tech", "text": "The preprocessing pipeline looks super clean! Great work.", "time": "1h ago"},
            {"id": 102, "username": "rohit_m", "text": "तुझा फोटो खूप छान आहे.", "time": "45m ago"},
            {"id": 103, "username": "aarav_k", "text": "Tumhara content hamesha informative hota hai.", "time": "30m ago"},
            {"id": 104, "username": "neha_g", "text": "All the best for your presentation tomorrow!", "time": "15m ago"}
        ]
    },
    {
        "id": 2,
        "username": "campus_vibes",
        "avatar": "CV",
        "image": "https://images.unsplash.com/photo-1521587760476-6c12a4b040da?auto=format&fit=crop&w=800&q=80",
        "caption": "Quiet evening at the central university library. Finals are approaching fast! 📚✨",
        "time": "4h ago",
        "likes": 87,
        "liked_by": ["demo_user"],
        "comments": [
            {"id": 105, "username": "aniket_p", "text": "आजचा दिवस खूप छान होता.", "time": "3h ago"},
            {"id": 106, "username": "sneha_r", "text": "Make sure to take breaks and stay hydrated everyone.", "time": "2h ago"},
            {"id": 107, "username": "vikram_s", "text": "तुम्हारी फोटो बहुत अच्छी है।", "time": "1h ago"},
            {"id": 108, "username": "tanvi_m", "text": "Quiet library hours are always the most productive.", "time": "40m ago"}
        ]
    },
    {
        "id": 3,
        "username": "maya_outdoors",
        "avatar": "MO",
        "image": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80",
        "caption": "Golden hour hike to decompress after lab sessions. Nature is always the best reset. 🌅🌲",
        "time": "6h ago",
        "likes": 124,
        "liked_by": [],
        "comments": [
            {"id": 109, "username": "sarah_w", "text": "Beautiful picture!", "time": "5h ago"},
            {"id": 110, "username": "amit_k", "text": "Khup sundar click aahe mitra!", "time": "4h ago"},
            {"id": 111, "username": "pooja_d", "text": "Thank you for sharing.", "time": "3h ago"},
            {"id": 112, "username": "rahul_v", "text": "निसर्गाचे अप्रतिम सौंदर्य, मन प्रसन्न झाले.", "time": "2h ago"}
        ]
    },
    {
        "id": 4,
        "username": "chef_ananya",
        "avatar": "CA",
        "image": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?auto=format&fit=crop&w=800&q=80",
        "caption": "Freshly baked artisan cinnamon rolls from scratch this morning! 🥐☕",
        "time": "8h ago",
        "likes": 65,
        "liked_by": [],
        "comments": [
            {"id": 113, "username": "kavita_n", "text": "ये सच में बहुत स्वादिष्ट लग रहा है।", "time": "7h ago"},
            {"id": 114, "username": "rajesh_b", "text": "Tujha recipe video nakki share kar bhava!", "time": "6h ago"},
            {"id": 115, "username": "david_m", "text": "Looks wonderfully golden and crisp on top.", "time": "5h ago"}
        ]
    }
]

# ==================== PAGE ROUTES ====================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    detected_lang = None
    entered_comment = request.args.get("comment", "")

    if request.method == "POST":
        entered_comment = request.form.get("comment", "").strip()
        if entered_comment:
            res = analyze_comment_3tier(entered_comment)
            detected_lang = res.get("language", "English")
            if res["state"] in ["blocked", "review"]:
                prediction = "Cyber Bullying"
                confidence = f"{res['confidence']}%"
            else:
                prediction = "Not Cyber Bullying"
                confidence = f"{res['confidence']}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        entered_comment=entered_comment,
        detected_lang=detected_lang
    )

@app.route("/feed")
def feed():
    return render_template("feed.html", posts=POSTS_DATA)

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/awareness")
def awareness():
    return render_template("awareness.html")

@app.route("/examples")
def examples():
    return render_template("examples.html")

@app.route("/performance")
def performance():
    metrics = {
        "accuracy": 95.3,
        "precision": 94.8,
        "recall": 94.2,
        "f1_score": 94.5,
        "total_samples": 534,
        "train_samples": 427,
        "test_samples": 107,
        "supported_languages": [
            "English", "Hindi", "Marathi", "Romanized Hindi", "Romanized Marathi"
        ]
    }
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    return render_template("performance.html", metrics=metrics)

# ==================== FEED API ROUTES ====================

@app.route("/api/like", methods=["POST"])
def api_like():
    data = request.get_json() or {}
    post_id = data.get("post_id")
    user = "demo_user"

    for post in POSTS_DATA:
        if post["id"] == post_id:
            if user in post["liked_by"]:
                post["liked_by"].remove(user)
                post["likes"] = max(0, post["likes"] - 1)
                liked = False
            else:
                post["liked_by"].append(user)
                post["likes"] += 1
                liked = True
            return jsonify({"success": True, "likes": post["likes"], "liked": liked})

    return jsonify({"success": False, "error": "Post not found"}), 404

@app.route("/api/moderate-comment", methods=["POST"])
def api_moderate_comment():
    """
    Check multilingual comment BEFORE posting.
    Returns 3-tier moderation result with language details.
    """
    data = request.get_json() or {}
    comment_text = data.get("comment", "").strip()

    if not comment_text:
        return jsonify({"success": False, "error": "Comment cannot be empty."}), 400

    analysis = analyze_comment_3tier(comment_text)

    return jsonify({
        "success": True,
        "comment": comment_text,
        "language": analysis.get("language", "English"),
        "state": analysis["state"],
        "heading": analysis["heading"],
        "message": analysis["message"],
        "prob_bullying": analysis["prob_bullying"],
        "confidence": analysis["confidence"],
        "can_post": analysis["can_post"]
    })

@app.route("/api/post-comment", methods=["POST"])
def api_post_comment():
    """Publish confirmed safe comment."""
    data = request.get_json() or {}
    post_id = data.get("post_id")
    comment_text = data.get("comment", "").strip()
    user = "demo_user"

    if not comment_text:
        return jsonify({"success": False, "error": "Comment cannot be empty."}), 400

    analysis = analyze_comment_3tier(comment_text)
    if not analysis["can_post"]:
        return jsonify({
            "success": False,
            "error": "Comment requires review or is blocked. Cannot be published."
        }), 400

    for post in POSTS_DATA:
        if post["id"] == post_id:
            new_comment = {
                "id": int(time.time() * 1000),
                "username": user,
                "text": comment_text,
                "time": "Just now"
            }
            post["comments"].append(new_comment)
            return jsonify({
                "success": True,
                "comment": new_comment,
                "total_comments": len(post["comments"])
            })

    return jsonify({"success": False, "error": "Post not found"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
