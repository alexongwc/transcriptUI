#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcription Quality Measurement using Unchunked Ground Truth

This script measures Word Error Rate (WER) and Character Error Rate (CER) of
transcription systems against an unchunked ground truth file, using HuggingFace's evaluate library.
"""

import os
import sys
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import pandas as pd

# File paths
SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTION_DIR = SCRIPT_DIR / "transcription_results"

FILES = {
    "ground_unchunked": str(TRANSCRIPTION_DIR / "1476_JunKaiOng_GEFA_m2_Transcript_Scored_PROPER_GroundTruth_Unchunked.xlsx"),
    "labs": str(TRANSCRIPTION_DIR / "1476_JunKaiOng_GEFA_m2_with_quality_11lab.xlsx"),
    "iflytek": str(TRANSCRIPTION_DIR / "segments_20250714_103125_iflytek.csv")
}

# Load HuggingFace metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Regular expressions for text cleaning
TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?")
BRACKET_RE = re.compile(r"\[.*?\]|\(.*?\)")
SPEAKER_RE = re.compile(r"^(?:Speaker\s*)?([A-Za-z]+(?:\s*Agent)?|MysteryShop(?:per)?):[\s]*", re.IGNORECASE)
PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")

# ---------------------------------------------------------------------------
# Text processing functions
# ---------------------------------------------------------------------------

def standardize_text(text: str) -> str:
    """Clean and standardize text for fair comparison."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # Remove timestamps (like 00:12:34)
    text = TIMESTAMP_RE.sub(" ", text)
    
    # Remove content in brackets or parentheses
    text = BRACKET_RE.sub(" ", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = PUNCTUATION_RE.sub(" ", text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text.strip()

def extract_speaker(text: str) -> Tuple[str, str]:
    """Extract speaker label from a line of text if present."""
    if not isinstance(text, str) or not text.strip():
        return ("", "")
        
    # Check if there's a speaker label
    match = SPEAKER_RE.search(text)
    if match:
        speaker = match.group(1)
        cleaned_text = text[match.end():].strip()
        return (speaker, cleaned_text)
    
    # No speaker found
    return ("", text.strip())

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_unchunked_groundtruth(path: str) -> pd.DataFrame:
    """Read unchunked ground truth file."""
    print(f"Reading unchunked ground truth from {path}")
    try:
        df = pd.read_excel(path, engine="openpyxl")
        
        if "Text" not in df.columns:
            print("WARNING: Expected column 'Text' not found in unchunked ground truth file")
            return pd.DataFrame(columns=["speaker", "text", "standardized_text"])
        
        # Standardize text for comparison
        df["standardized_text"] = df["Text"].astype(str).apply(standardize_text)
        
        # Make sure speaker column exists
        if "Speaker" not in df.columns:
            df["Speaker"] = ""
            
        # Rename columns for consistency
        df = df.rename(columns={"Speaker": "speaker", "Text": "text"})
        
        print(f"Unchunked ground truth: {len(df)} segments loaded")
        return df
        
    except Exception as e:
        print(f"ERROR reading unchunked ground truth file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["speaker", "text", "standardized_text"])

def read_11lab(path: str) -> pd.DataFrame:
    """Read 11lab file and extract speaker segments."""
    print(f"Reading 11lab from {path}")
    try:
        df = pd.read_excel(path, engine="openpyxl")
        if "Text" not in df.columns or "Start Time" not in df.columns:
            print("WARNING: Expected columns not found in 11lab file")
            return pd.DataFrame(columns=["speaker", "text", "standardized_text", "start_time"])
        
        # Process each row to extract speakers and text
        segments = []
        
        for idx, row in df.iterrows():
            text = str(row["Text"])
            # Extract timestamp and convert to standard format
            start_time = str(row["Start Time"]).replace(',', '.').strip()
            
            # Extract speaker if present
            speaker, content = extract_speaker(text)
            
            segments.append({
                "row_index": idx,
                "speaker": speaker,
                "text": content if content else text,  # Use extracted content if speaker found
                "standardized_text": standardize_text(content if content else text),
                "start_time": start_time
            })
        
        # Create DataFrame
        result_df = pd.DataFrame(segments)
        print(f"11lab: {len(result_df)} segments processed")
        return result_df
        
    except Exception as e:
        print(f"ERROR reading 11lab file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["speaker", "text", "standardized_text", "start_time"])

def read_iflytek(path: str) -> pd.DataFrame:
    """Read iflytek CSV file and extract speaker segments."""
    print(f"Reading iflytek from {path}")
    try:
        df = pd.read_csv(path)
        if "Text" not in df.columns or "Timestamp" not in df.columns:
            print("WARNING: Expected columns not found in iflytek file")
            return pd.DataFrame(columns=["speaker", "text", "standardized_text", "start_time"])
        
        # Process each row to extract speakers and text
        segments = []
        
        for idx, row in df.iterrows():
            text = str(row["Text"])
            # Get timestamp (already in standard format)
            start_time = str(row["Timestamp"])
            
            # Extract speaker if present
            speaker, content = extract_speaker(text)
            # Also check if Speaker column exists
            if "Speaker" in df.columns and pd.notna(row["Speaker"]):
                row_speaker = str(row["Speaker"]).strip()
                if row_speaker and not speaker:  # Prefer extracted speaker if both exist
                    speaker = row_speaker
                    
            segments.append({
                "row_index": idx,
                "speaker": speaker,
                "text": content if content else text,  # Use extracted content if speaker found
                "standardized_text": standardize_text(content if content else text),
                "start_time": start_time
            })
        
        # Create DataFrame
        result_df = pd.DataFrame(segments)
        print(f"iflytek: {len(result_df)} segments processed")
        return result_df
        
    except Exception as e:
        print(f"ERROR reading iflytek file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["speaker", "text", "standardized_text", "start_time"])

# ---------------------------------------------------------------------------
# Alignment and evaluation functions
# ---------------------------------------------------------------------------

def align_segments_by_speaker(ground_df, transcription_df, time_based=False):
    """Align ground truth segments with transcription segments.
    
    If time_based is True, uses timestamps to align segments.
    Otherwise uses speaker labels and text similarity.
    
    Returns aligned segments as (ground_segment, transcription_segment) pairs.
    """
    aligned_pairs = []
    
    # Group transcription segments by speaker
    speakers = {}
    for _, row in transcription_df.iterrows():
        speaker = row["speaker"]
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(row)
    
    # For each ground truth segment, find best matching transcription segment
    for _, ground_row in ground_df.iterrows():
        ground_speaker = ground_row["speaker"]
        ground_text = ground_row["standardized_text"]
        
        # Skip empty segments
        if not ground_text.strip():
            continue
        
        # Try to match by speaker first if speaker exists
        matching_segments = []
        
        if ground_speaker and ground_speaker in speakers:
            matching_segments = speakers[ground_speaker]
        else:
            # If no matching speaker, consider all segments
            matching_segments = [row for spk_rows in speakers.values() for row in spk_rows]
            
        if not matching_segments:
            continue
            
        # Find best match by text similarity (we could use more sophisticated algorithms here)
        best_match = None
        best_score = -1
        
        for trans_row in matching_segments:
            trans_text = trans_row["standardized_text"]
            # Simple word overlap score (can be improved)
            ground_words = set(ground_text.split())
            trans_words = set(trans_text.split())
            if not ground_words or not trans_words:
                continue
                
            overlap = len(ground_words.intersection(trans_words))
            score = overlap / max(len(ground_words), len(trans_words))
            
            if score > best_score:
                best_score = score
                best_match = trans_row
        
        # Only keep matches with reasonable similarity
        if best_match is not None and best_score > 0.1:  # Threshold can be adjusted
            aligned_pairs.append((ground_row, best_match))
    
    print(f"Aligned {len(aligned_pairs)} segments between ground truth and transcription")
    return aligned_pairs

def compute_metrics_on_aligned(aligned_pairs):
    """Compute WER and CER on aligned segment pairs."""
    if not aligned_pairs:
        print("WARNING: No aligned segments found")
        return {"wer": 1.0, "cer": 1.0, "segment_count": 0}  # 100% error if no aligned segments
    
    references = []
    predictions = []
    
    for ground_row, trans_row in aligned_pairs:
        references.append(ground_row["standardized_text"])
        predictions.append(trans_row["standardized_text"])
    
    # Compute metrics
    wer_score = wer_metric.compute(references=references, predictions=predictions)
    cer_score = cer_metric.compute(references=references, predictions=predictions)
    
    return {
        "wer": wer_score,
        "cer": cer_score,
        "segment_count": len(aligned_pairs)
    }

def compute_metrics_combined(ground_df, transcription_df):
    """Compute metrics on combined text (all segments joined together)."""
    # Combine all text into single strings
    ref_text = " ".join(ground_df["standardized_text"].dropna())
    hyp_text = " ".join(transcription_df["standardized_text"].dropna())
    
    # Guard against empty text
    if not ref_text or not hyp_text:
        print("WARNING: Empty reference or hypothesis text")
        return {"wer": 1.0, "cer": 1.0}  # 100% error if empty
    
    # Compute metrics
    wer_score = wer_metric.compute(references=[ref_text], predictions=[hyp_text])
    cer_score = cer_metric.compute(references=[ref_text], predictions=[hyp_text])
    
    return {"wer": wer_score, "cer": cer_score, "combined": True}

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def save_results_to_file(results, output_file="wer_cer_results.txt"):
    """Save metrics results to a text file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Write header
            f.write("=== Transcription Quality Measurement Results ===\n\n")
            
            # Per-Segment Alignment Results
            f.write("--- Per-Segment Alignment Results ---\n")
            
            f.write(f"11lab: {results['labs_aligned']['segment_count']} segments aligned\n")
            f.write(f"  Word Error Rate: {results['labs_aligned']['wer']*100:.2f}%\n")
            f.write(f"  Char Error Rate: {results['labs_aligned']['cer']*100:.2f}%\n\n")
            
            f.write(f"iflytek: {results['iflytek_aligned']['segment_count']} segments aligned\n")
            f.write(f"  Word Error Rate: {results['iflytek_aligned']['wer']*100:.2f}%\n")
            f.write(f"  Char Error Rate: {results['iflytek_aligned']['cer']*100:.2f}%\n\n")
            
            # Combined Text Results
            f.write("--- Combined Text Results ---\n")
            
            f.write(f"11lab:\n")
            f.write(f"  Word Error Rate: {results['labs_combined']['wer']*100:.2f}%\n")
            f.write(f"  Char Error Rate: {results['labs_combined']['cer']*100:.2f}%\n\n")
            
            f.write(f"iflytek:\n")
            f.write(f"  Word Error Rate: {results['iflytek_combined']['wer']*100:.2f}%\n")
            f.write(f"  Char Error Rate: {results['iflytek_combined']['cer']*100:.2f}%\n\n")
            
            # Best system
            best = "11lab" if results['labs_aligned']['wer'] < results['iflytek_aligned']['wer'] else "iflytek"
            f.write(f"Best transcription system: {best} (lower WER)\n\n")
            
            # Notes
            f.write("Note: Metrics are calculated after standardizing all texts (removing speaker labels,\n")
            f.write("timestamps, punctuation, etc.) and comparing using speaker segment alignment.\n")
            f.write("Per-Segment Alignment metrics are more accurate as they match corresponding\n")
            f.write("segments between ground truth and transcription systems.\n")
        
        print(f"\nResults saved to {output_file}")
        return True
    except Exception as e:
        print(f"\nERROR saving results to file: {e}")
        return False


def main():
    try:
        # Print file paths
        print("\n=== Processing Files ===")
        print(f"Unchunked Ground truth: {FILES['ground_unchunked']}")
        print(f"11lab: {FILES['labs']}")
        print(f"iflytek: {FILES['iflytek']}")
        
        # Check files exist
        for path in FILES.values():
            if not os.path.exists(path):
                sys.exit(f"File not found: {path}")

        # Read and process files
        print("\n=== Reading and Processing Files ===")
        ground_df = read_unchunked_groundtruth(FILES['ground_unchunked'])
        labs_df = read_11lab(FILES['labs'])
        iflytek_df = read_iflytek(FILES['iflytek'])

        # Show alignment counts
        print("\n=== Aligning Speaker Segments ===")
        labs_aligned = align_segments_by_speaker(ground_df, labs_df)
        iflytek_aligned = align_segments_by_speaker(ground_df, iflytek_df)
        
        # Compute metrics on aligned pairs
        print("\n=== Computing Metrics on Aligned Segments ===")
        labs_aligned_metrics = compute_metrics_on_aligned(labs_aligned)
        iflytek_aligned_metrics = compute_metrics_on_aligned(iflytek_aligned)
        
        # Also compute metrics on combined text for comparison
        print("\n=== Computing Metrics on Combined Text ===")
        labs_combined_metrics = compute_metrics_combined(ground_df, labs_df)
        iflytek_combined_metrics = compute_metrics_combined(ground_df, iflytek_df)
        
        # Collect all results
        all_results = {
            "labs_aligned": labs_aligned_metrics,
            "iflytek_aligned": iflytek_aligned_metrics,
            "labs_combined": labs_combined_metrics,
            "iflytek_combined": iflytek_combined_metrics
        }
        
        # Save results to file
        save_results_to_file(all_results)
        
        # Print a simple summary that's less likely to get corrupted in output
        print("\n==== SUMMARY OF KEY METRICS =====")
        print(f"11LAB WER: {labs_aligned_metrics['wer']*100:.2f}% CER: {labs_aligned_metrics['cer']*100:.2f}%")
        print(f"IFLYTEK WER: {iflytek_aligned_metrics['wer']*100:.2f}% CER: {iflytek_aligned_metrics['cer']*100:.2f}%")
        
        # Print results with nice formatting
        print("\n=== Transcription Error Rates (Per-Segment Alignment) ===")
        print(f"11lab: {labs_aligned_metrics['segment_count']} segments aligned")
        print(f"  Word Error Rate: {labs_aligned_metrics['wer']*100:.2f}%")
        print(f"  Char Error Rate: {labs_aligned_metrics['cer']*100:.2f}%")
        print()
        print(f"iflytek: {iflytek_aligned_metrics['segment_count']} segments aligned")
        print(f"  Word Error Rate: {iflytek_aligned_metrics['wer']*100:.2f}%")
        print(f"  Char Error Rate: {iflytek_aligned_metrics['cer']*100:.2f}%")
        
        print("\n=== Transcription Error Rates (Combined Text) ===")
        print(f"11lab:")
        print(f"  Word Error Rate: {labs_combined_metrics['wer']*100:.2f}%")
        print(f"  Char Error Rate: {labs_combined_metrics['cer']*100:.2f}%")
        print()
        print(f"iflytek:")
        print(f"  Word Error Rate: {iflytek_combined_metrics['wer']*100:.2f}%")
        print(f"  Char Error Rate: {iflytek_combined_metrics['cer']*100:.2f}%")
        
        # Recommend which is better based on aligned metrics
        best = "11lab" if labs_aligned_metrics["wer"] < iflytek_aligned_metrics["wer"] else "iflytek"
        print(f"\nBest transcription system: {best} (lower WER)")
        
        print("\nNote: Metrics are calculated after standardizing all texts (removing speaker labels,")
        print("timestamps, punctuation, etc.) and comparing using speaker segment alignment.")
        print("Per-Segment Alignment metrics are more accurate as they match corresponding")
        print("segments between ground truth and transcription systems.")
    
    except Exception as e:
        print(f"\nERROR: An exception occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
