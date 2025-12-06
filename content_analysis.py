"""
content_analysis.py
Pipeline 2: ASR + Content and Business Logic Analysis

Outputs:
 - transcript (string)
 - business_score (0-100)
 - content_evals (dict of problem_clarity, product_diff, business_model, market, revenue_logic, competition)
 - structure_detected (hook, problem, solution, ask) booleans
"""

from jiwer import wer
# import a whisper or HF ASR wrapper - keep modular for user choice
# We'll provide a wrapper function that tries to use openai/whisper/transformers if available.

def transcribe_with_whisper(audio_path, model_name="openai/whisper-small"):
    """
    Placeholder: user may replace with direct whisper or google ASR call.
    If they have 'whisper' locally installed, use that. Otherwise use HF pipeline.
    For now this function returns a stub to let the rest of pipeline run offline.
    """
    try:
        # Attempt to use transformers pipeline if available
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model=model_name)
        res = asr(audio_path)
        return res.get("text", "")
    except Exception:
        # Fallback: return empty string - user should plug in ASR
        return ""

def evaluate_content(transcript):
    """
    Use simple heuristic / keyword checks and shallow NLP to score different content axes.
    More advanced approach: fine-tuned classifiers per axis.
    Each subscore is 0-1 (we later scale to 0-100).
    """
    if not transcript:
        empty = {k:0.0 for k in ["problem_clarity","product_diff","business_model","market","revenue_logic","competition_awareness"]}
        return empty

    txt = transcript.lower()

    def keyword_score(keywords, weight=1.0):
        found = 0
        for k in keywords:
            if k in txt:
                found += 1
        return min(1.0, found / max(1, len(keywords)))

    # heuristics: these lists can be expanded
    problem_kws = ["problem","pain","issue","need","struggle"]
    product_kws = ["differenti", "unique", "patent", "proprietary", "better than", "vs", "unlike"]
    model_kws = ["subscription","revenue","monetize","pricing","pay","fee","commission","margins"]
    market_kws = ["market","customers","target","segment","tam","sam","som","growth","demand"]
    revenue_kws = ["price","pricing","avg revenue","lifetime value","ltv","revenue"]
    competition_kws = ["competitor","competition","others","alternatives","vs","compare"]

    subscores = {
        "problem_clarity": keyword_score(problem_kws),
        "product_diff": keyword_score(product_kws),
        "business_model": keyword_score(model_kws),
        "market": keyword_score(market_kws),
        "revenue_logic": keyword_score(revenue_kws),
        "competition_awareness": keyword_score(competition_kws)
    }
    return subscores

def detect_structure(transcript):
    """
    Simple presence-based detection for Hook→Problem→Solution→Ask
    """
    if not transcript:
        return {"hook": False, "problem": False, "solution": False, "ask": False}
    t = transcript.lower()
    return {
        "hook": ("today i" in t) or ("imagine" in t) or ("picture this" in t) or ("did you know" in t),
        "problem": ("problem" in t) or ("pain" in t) or ("issue" in t),
        "solution": ("solution" in t) or ("we provide" in t) or ("our product" in t) or ("we built" in t),
        "ask": ("we are raising" in t) or ("ask" in t) or ("invest" in t) or ("seeking" in t) or ("looking for" in t)
    }

def compute_business_score(subscores, structure):
    # Weighted average of subscores plus bonus for having structure
    weights = {
        "problem_clarity": 0.18,
        "product_diff": 0.18,
        "business_model": 0.18,
        "market": 0.18,
        "revenue_logic": 0.14,
        "competition_awareness": 0.14
    }
    score = 0.0
    for k,w in weights.items():
        score += subscores.get(k,0.0)*w
    # structure bonus
    structure_bonus = sum([1 for k,v in structure.items() if v]) / 4.0  # 0-1
    score = score * 0.9 + 0.1 * structure_bonus
    return float(round(min(1.0, score) * 100.0, 2))
