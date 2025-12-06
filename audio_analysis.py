"""
audio_analysis.py
Functions for Pipeline 1: Voice & Tone Analysis

Outputs:
 - features dict (pitch, tempo, volume_variation, pauses_count, filler_count, hesitation_index)
 - tone_score (0-100)
 - tone_tags (e.g., confident, nervous, monotone, enthusiastic)
"""

import numpy as np
import librosa
import re
from collections import Counter

FILLER_WORDS = {"um","uh","like","so","you know","actually","basically","right","I mean"}

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_basic_features(y, sr):
    # pitch (f0) via librosa.pyin
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                      fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0)
        median_f0 = float(np.median(f0[f0>0])) if np.any(f0>0) else 0.0
        f0_std = float(np.std(f0))
    except Exception:
        median_f0 = 0.0
        f0_std = 0.0

    # tempo (beats per minute) approximates speech rate
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # energy
    S, _ = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S).mean()
    # volume variation: std of short-time energy
    frame_length = 1024
    hop_length = 512
    env = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    vol_var = float(np.std(env) / (np.mean(env)+1e-9))

    return {
        "median_f0": median_f0,
        "f0_std": f0_std,
        "tempo": float(tempo),
        "rms": float(rms),
        "volume_variation": vol_var
    }

def detect_pauses(y, sr, top_db=30, min_silence_len=0.25):
    # approximate pauses by splitting non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=512)
    total_duration = len(y)/sr
    speech_dur = sum([(end-start)/sr for start,end in intervals])
    pauses = max(0, total_duration - speech_dur)
    # count of pauses by thresholding silent gaps
    gap_count = 0
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            gap = (intervals[i][0] - intervals[i-1][1]) / sr
            if gap >= min_silence_len:
                gap_count += 1
    return {"total_pauses_seconds": pauses, "pause_count": gap_count, "speech_duration": speech_dur}

def detect_fillers_from_transcript(transcript):
    """
    Naive filler detection from transcript text (lowercased).
    Should be called after ASR (pipeline 2). Returns count and list of top fillers.
    """
    if not transcript:
        return {"filler_count": 0, "top_fillers": []}
    text = transcript.lower()
    counts = Counter()
    for fw in FILLER_WORDS:
        pattern = r"\b" + re.escape(fw) + r"\b"
        found = re.findall(pattern, text)
        if found:
            counts[fw] += len(found)
    top = counts.most_common()
    return {"filler_count": sum(counts.values()), "top_fillers": top}

def compute_hesitation_index(pauses_info, speech_duration):
    # hesitation index = pause seconds per speech minute
    speech_minutes = max(0.001, speech_duration/60.0)
    hesitation_index = pauses_info["total_pauses_seconds"] / speech_minutes
    return float(hesitation_index)

def score_tone(features, pauses_info, filler_count):
    """
    Combine normalized sub-scores into a 0-100 tone score.
    Subscores: clarity (tempo within [110,180] for speech? we will heuristically map),
    energy (volume variation), confidence (f0_std high = expressive; low = monotone),
    hesitation penalty (more hesitation lowers score), filler penalty.
    """
    # clarity: tempo mapped
    tempo = features.get("tempo", 120.0)
    # map tempo (spoken words per minute roughly) - we use heuristic: ideal 120-180
    clarity = 1.0 - abs(tempo - 150)/150.0
    clarity = np.clip(clarity, 0.0, 1.0)

    # energy from volume variation
    energy = np.tanh(features.get("volume_variation", 0.1)*5.0)  # 0-1

    # expressiveness: f0_std
    f0_std = features.get("f0_std", 0.0)
    express = np.tanh(f0_std/30.0)

    # hesitation penalty
    hesitation_index = compute_hesitation_index(pauses_info, pauses_info.get("speech_duration",1.0))
    hes_pen = np.exp(-0.05 * hesitation_index)  # decays with more hesitation

    # filler penalty
    filler_pen = np.exp(-0.3 * filler_count)

    raw_score = (0.35*clarity + 0.25*energy + 0.25*express + 0.15*hes_pen) * filler_pen
    tone_score = float(np.clip(raw_score * 100.0, 0.0, 100.0))

    # tags heuristics
    tags = []
    if tone_score > 75:
        tags.append("confident")
    elif tone_score > 50:
        tags.append("adequate")
    else:
        tags.append("nervous")
    if express < 0.2:
        tags.append("monotone")
    if filler_count > 3:
        tags.append("many_fillers")
    if hesitation_index > 10:
        tags.append("high_hesitation")

    return {"tone_score": tone_score, "tags": tags, "subscores": {
        "clarity": float(clarity*100),
        "energy": float(energy*100),
        "expressiveness": float(express*100),
        "hesitation_index": hesitation_index,
        "filler_count": filler_count
    }}
