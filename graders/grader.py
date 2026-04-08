from typing import List

def grade_classification(predicted: str, actual: str) -> float:
    return 1.0 if predicted.lower() == actual.lower() else 0.0

def grade_priority(predicted: str, actual: str) -> float:
    levels = {"low": 1, "medium": 2, "high": 3}
    p = levels.get(predicted.lower(), 0)
    a = levels.get(actual.lower(), 1)
    return max(0.0, 1.0 - abs(p - a) * 0.5)

def grade_response(response_text: str, required_keywords: List[str]) -> float:
    if not response_text or len(response_text.strip()) < 5:
        return 0.0

    text = response_text.lower()
    text_words = set(text.split())
    
    # 🟠 Anti-Exploit Strategy: Set-based intersection matching 
    # Prevents "keyword stuffing" (repeating words to hit triggers)
    hit_count = 0
    for kw in required_keywords:
        kw_words = set(kw.lower().split())
        if kw_words.intersection(text_words):
            hit_count += 1
    
    keyword_score = hit_count / max(1, len(required_keywords))
    
    # Length Normalization & Penalty for reward hacking
    word_count = len(text_words)
    length_penalty = 1.0
    if word_count > 80: # Reward hacking detection (stuffing text)
        length_penalty = 0.8
    elif word_count < 10:
        length_penalty = 0.5
        
    keyword_score *= length_penalty

    # Semantic Intent Signal
    intent_words = ["confirm", "schedule", "discuss", "update", "regards", "best", "thanks"]
    intent_hits = sum(1 for word in intent_words if word in text)
    intent_score = min(intent_hits / 3, 1.0) # Need 3 intent markers for full signal

    structure_score = 1.0 if "." in response_text else 0.5

    return round(
        0.4 * keyword_score +
        0.2 * intent_score +
        0.2 * structure_score +
        0.2 * min(len(response_text) / 200, 1.0),
        2
    )
