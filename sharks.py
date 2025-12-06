"""
sharks.py
Generate persona-driven feedback based on tone_score, business_score and transcript.

This module offers two approaches:
 - template_feedback: Generate feedback using deterministic templates (no external API).
 - llm_feedback: Generate feedback using an LLM (OpenAI or HF) given a prompt.

By default, code uses template_feedback so the project runs offline.
"""

import random
from textwrap import fill

def format_block(title, body):
    return f"**{title}:**\n{body}\n"

# define shark personas
SHARKS = [
    {
        "id": "visionary",
        "name": "The Visionary",
        "focus": ["market","innovation"],
        "tone": "encouraging, forward-looking"
    },
    {
        "id": "finance",
        "name": "The Finance Shark",
        "focus": ["revenue","margins"],
        "tone": "direct, numbers-focused"
    },
    {
        "id": "skeptic",
        "name": "The Skeptic",
        "focus": ["assumptions","risks"],
        "tone": "challenging, blunt"
    },
    {
        "id": "customer",
        "name": "The Customer Advocate",
        "focus": ["problem","user"],
        "tone": "empathetic, user-first"
    }
]

def template_shark_feedback(shark, tone_score, business_score, transcript, content_subscores):
    # Create short strengths & weaknesses & suggestion
    strengths = []
    weaknesses = []
    suggestions = []
    # tonal comments
    if tone_score > 75:
        strengths.append("delivery was confident and energetic")
    elif tone_score > 50:
        strengths.append("delivery was adequate, but could be more energetic")
    else:
        weaknesses.append("delivery felt nervous or lacked clarity")

    # business comments guided by the shark's focus
    if "market" in shark["focus"] or "innovation" in shark["focus"]:
        if content_subscores.get("market",0) > 0.5:
            strengths.append("market opportunity is described well")
        else:
            weaknesses.append("market sizing/opportunity is vague")
        if content_subscores.get("product_diff",0) > 0.4:
            strengths.append("product appears differentiated")
        else:
            weaknesses.append("product differentiation is weak or not spelled out")

    if "revenue" in shark["focus"] or "margins" in shark["focus"]:
        if content_subscores.get("business_model",0) > 0.5 or content_subscores.get("revenue_logic",0) > 0.5:
            strengths.append("business model and revenue logic are present")
        else:
            weaknesses.append("unclear revenue model; need unit economics")

    if "assumptions" in shark["focus"]:
        if content_subscores.get("competition_awareness",0) < 0.4:
            weaknesses.append("competitive landscape seems under-explored")
        if content_subscores.get("problem_clarity",0) < 0.4:
            weaknesses.append("user problem needs stronger validation")

    if "problem" in shark["focus"] or "user" in shark["focus"]:
        if content_subscores.get("problem_clarity",0) > 0.5:
            strengths.append("problem statement is compelling")
        else:
            weaknesses.append("problem statement is not clear enough for users")

    # make sentences
    s = strengths[:2] or ["some points are promising"]
    w = weaknesses[:3] or ["main risks were not highlighted"]
    sug = suggestions or ["Provide more concrete numbers, user evidence, and a clear ask."]

    body_lines = []
    body_lines.append("Strengths: " + ", ".join(s) + ".")
    body_lines.append("Weaknesses: " + ", ".join(w) + ".")
    body_lines.append("Recommendation: " + sug[0])

    # include short score summary
    body_lines.append(f"(Tone score: {tone_score:.1f} / 100, Business score: {business_score:.1f} / 100)")

    body = "\n".join(body_lines)
    # wrap nicely
    body = fill(body, width=100)
    return format_block(shark["name"], body)

def generate_panel_feedback(tone_result, business_result, transcript, use_llm=False, llm_func=None):
    """
    tone_result: dict from audio_analysis.score_tone
    business_result: dict with keys 'business_score', 'subscores', 'structure'
    """
    tone_score = tone_result["tone_score"]
    subs = business_result.get("subscores", {})
    business_score = business_result.get("business_score", 0.0)
    # choose 3 sharks (or all)
    chosen = SHARKS[:3]
    blocks = []
    for shark in chosen:
        if use_llm and llm_func:
            # llm_func should accept (persona, tone_score, business_score, transcript, subs) and return text
            text = llm_func(shark, tone_score, business_score, transcript, subs)
            blocks.append(format_block(shark["name"], text))
        else:
            blocks.append(template_shark_feedback(shark, tone_score, business_score, transcript, subs))
    # final recommendation (simple rule)
    if business_score > 70 and tone_score > 60:
        final = "INVEST"
    elif business_score > 50 and tone_score > 45:
        final = "NEED MORE INFO"
    else:
        final = "NOT INVEST"

    combined = "\n\n".join(blocks)
    combined += "\n\n### Final Recommendation: " + final
    return combined
