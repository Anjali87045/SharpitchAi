"""
main.py
Demo harness to run the two pipelines and produce shark feedback.

Usage:
 python main.py --audio path/to/pitch.wav --use-llm 0
"""

import argparse
from audio_analysis import load_audio, extract_basic_features, detect_pauses, detect_fillers_from_transcript, score_tone
from content_analysis import transcribe_with_whisper, evaluate_content, detect_structure, compute_business_score
from sharks import generate_panel_feedback

def run_pipeline(audio_path, use_llm=False):
    print("Loading audio...")
    y, sr = load_audio(audio_path)
    print("Extracting features...")
    features = extract_basic_features(y, sr)
    pauses_info = detect_pauses(y, sr)
    # transcribe
    print("Transcribing (ASR)...")
    transcript = transcribe_with_whisper(audio_path)
    print("Transcript length:", len(transcript))
    filler_info = detect_fillers_from_transcript(transcript)
    # tone scoring
    tone_result = score_tone(features, pauses_info, filler_info.get("filler_count", 0))
    # content analysis
    subscores = evaluate_content(transcript)
    structure = detect_structure(transcript)
    business_score = compute_business_score(subscores, structure)
    business_result = {"business_score": business_score, "subscores": subscores, "structure": structure}
    # Shark panel
    panel = generate_panel_feedback(tone_result, {"business_score": business_score, "subscores": subscores}, transcript, use_llm=use_llm)
    # show outputs
    print("\n=== ANALYSIS SUMMARY ===")
    print("Tone result:", tone_result)
    print("Business result:", business_result)
    print("\n=== SHARK PANEL ===\n")
    print(panel)
    return {
        "tone_result": tone_result,
        "business_result": business_result,
        "transcript": transcript,
        "panel": panel
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to wav/mp3 audio file")
    parser.add_argument("--use-llm", type=int, default=0, help="Use LLM for feedback generation? 1=yes")
    args = parser.parse_args()
    run_pipeline(args.audio, use_llm=bool(args.use_llm))
