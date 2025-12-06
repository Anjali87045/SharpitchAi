"""
llm_feedback.py
Example LLM wrapper using OpenAI. Put your OPENAI_API_KEY in .env.

If you don't want to use an API, skip this file: the template-based feedback is solid.
"""

import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are acting as {persona_name}, an investor with the following style: {persona_style}.
Given the pitch transcript below, the speaker got a tone score of {tone_score:.1f} and a business score of {business_score:.1f}.
Provide a short (3-5 bullet) feedback:
 - one short summary sentence
 - two strengths (concise)
 - two weaknesses (concise)
 - one specific actionable suggestion

Transcript:
{transcript}

Return the response in plain text. Be concise.
"""

def llm_func(shark, tone_score, business_score, transcript, subscores):
    prompt = PROMPT_TEMPLATE.format(
        persona_name=shark["name"],
        persona_style=shark["tone"],
        tone_score=tone_score,
        business_score=business_score,
        transcript=transcript[:3000]  # truncate to safe length
    )
    resp = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    return resp.choices[0].text.strip()
