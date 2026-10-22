"""
Automated Multilingual Moderation Test Suite
Domain: NLP Text Classification
Languages Tested: English, Hindi, Marathi, Romanized Hindi, Romanized Marathi
Verifies: 3-tier outcomes (safe, review, blocked), language detection, and policy safeguards
"""

import sys
import os

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import analyze_comment_3tier, detect_language, clean_text

TEST_CASES = [
    # 1. Safe English comment
    {
        "id": "EN-SAFE",
        "text": "Beautiful picture!",
        "expected_lang": "English",
        "expected_states": ["safe"],
        "description": "Safe English compliment"
    },
    # 2. Harmful English comment
    {
        "id": "EN-HARMFUL",
        "text": "You are useless and nobody likes you.",
        "expected_lang": "English",
        "expected_states": ["blocked"],
        "description": "Harmful English personal insult and harassment"
    },
    # 3. Safe Hindi comment
    {
        "id": "HI-SAFE",
        "text": "तुम्हारी फोटो बहुत अच्छी है।",
        "expected_lang": "Hindi",
        "expected_states": ["safe"],
        "description": "Safe Hindi photo appreciation in Devanagari"
    },
    # 4. Harmful Hindi comment
    {
        "id": "HI-HARMFUL",
        "text": "तुम किसी काम के नहीं हो।",
        "expected_lang": "Hindi",
        "expected_states": ["blocked"],
        "description": "Harmful Hindi humiliation in Devanagari"
    },
    # 5. Safe Marathi comment
    {
        "id": "MR-SAFE",
        "text": "तुझा फोटो खूप छान आहे.",
        "expected_lang": "Marathi",
        "expected_states": ["safe"],
        "description": "Safe Marathi photo compliment in Devanagari"
    },
    # 6. Harmful Marathi comment
    {
        "id": "MR-HARMFUL",
        "text": "तू काहीच कामाचा नाहीस.",
        "expected_lang": "Marathi",
        "expected_states": ["blocked"],
        "description": "Harmful Marathi insult in Devanagari"
    },
    # 7. Romanized Hindi comment (Safe & Harmful)
    {
        "id": "ROM-HI-SAFE",
        "text": "Tum bahut achhe ho.",
        "expected_lang": "Romanized Hindi",
        "expected_states": ["safe"],
        "description": "Safe Romanized Hindi friendly compliment"
    },
    {
        "id": "ROM-HI-HARMFUL",
        "text": "Tum kisi kaam ke nahi ho bilkul bewakoof.",
        "expected_lang": "Romanized Hindi",
        "expected_states": ["blocked"],
        "description": "Harmful Romanized Hindi insult"
    },
    # 8. Romanized Marathi comment (Safe & Harmful)
    {
        "id": "ROM-MR-SAFE",
        "text": "Tujha photo khup chhan aahe.",
        "expected_lang": "Romanized Marathi",
        "expected_states": ["safe"],
        "description": "Safe Romanized Marathi photo appreciation"
    },
    {
        "id": "ROM-MR-HARMFUL",
        "text": "Tu kahich kaamacha nahis bindok manus.",
        "expected_lang": "Romanized Marathi",
        "expected_states": ["blocked"],
        "description": "Harmful Romanized Marathi insult"
    },
    # 9. Respectful disagreement
    {
        "id": "DISAGREE-EN",
        "text": "I disagree with your opinion.",
        "expected_lang": "English",
        "expected_states": ["safe"],
        "description": "Polite English disagreement (Not Cyberbullying)"
    },
    {
        "id": "DISAGREE-MR",
        "text": "मला तुझं मत मान्य नाही.",
        "expected_lang": "Marathi",
        "expected_states": ["safe"],
        "description": "Polite Marathi disagreement (Not Cyberbullying)"
    },
    # 10. Ordinary advice
    {
        "id": "ADVICE-EN",
        "text": "Please take care of yourself.",
        "expected_lang": "English",
        "expected_states": ["safe"],
        "description": "Caring healthcare suggestion (Safe)"
    },
    {
        "id": "ADVICE-HINGLISH",
        "text": "See as a doctor I suggest riya ki har bar doctors change mt kro ek hi doctor k pass unk prescribed kre huye medicines ko phle khatam kro thk tarike se",
        "expected_lang": "Romanized Hindi",
        "expected_states": ["safe"],
        "description": "Medical prescription advice in Hinglish (Safe)"
    },
    # 11. Negative but non-abusive criticism vs derogatory critique
    {
        "id": "CRITIQUE-MILD",
        "text": "The audio quality is slightly low in the second half.",
        "expected_lang": "English",
        "expected_states": ["safe", "review"],
        "description": "Constructive technical feedback"
    },
    {
        "id": "CRITIQUE-DEROGATORY",
        "text": "very bad picture",
        "expected_lang": "English",
        "expected_states": ["review", "blocked"],
        "description": "Derogatory media critique (Flagged for review or blocked)"
    }
]

def run_tests():
    print("=" * 75)
    print("  MULTILINGUAL CYBER BULLYING MODERATION TEST RUNNER")
    print("=" * 75)
    passed = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        text = case["text"]
        result = analyze_comment_3tier(text)
        detected_lang = result.get("language")
        state = result.get("state")
        prob = result.get("prob_bullying", 0.0)
        conf = result.get("confidence", 0)

        lang_match = (detected_lang == case["expected_lang"])
        state_match = (state in case["expected_states"])

        status_str = "PASS" if (lang_match and state_match) else "WARN/FAIL"
        if lang_match and state_match:
            passed += 1

        print(f"\n[{status_str}] Test ID: {case['id']}")
        print(f"       Description  : {case['description']}")
        print(f"       Comment Text : \"{text}\"")
        print(f"       Cleaned Text : \"{clean_text(text)}\"")
        print(f"       Language     : Detected = {detected_lang} (Expected = {case['expected_lang']})")
        print(f"       Outcome State: {state.upper()} (Expected one of = {[s.upper() for s in case['expected_states']]})")
        print(f"       Probability  : {prob} | Confidence: {conf}%")
        print(f"       Heading      : {result.get('heading')}")

    print("\n" + "=" * 75)
    print(f"  TEST SUMMARY: {passed}/{total} Tests Passed Successfully! ({passed/total*100:.1f}%)")
    print("=" * 75)

    if passed == total:
        print("\nAll multilingual moderation test cases verified and compliant!")
        return 0
    else:
        print(f"\n{total - passed} test(s) had variations. Review details above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
