#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ground Truth Unchunker

This script extracts individual speaker utterances from chunked ground truth files,
creating a new Excel file where each row represents a single speaker's utterance.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

# Default paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTION_DIR = SCRIPT_DIR / "transcription_results"
INPUT_FILE = TRANSCRIPTION_DIR / "1476_JunKaiOng_GEFA_m2_Transcript_Scored_PROPER_GroundTruth.xlsx"
OUTPUT_FILE = TRANSCRIPTION_DIR / "1476_JunKaiOng_GEFA_m2_Transcript_Scored_PROPER_GroundTruth_Unchunked.xlsx"

# Speaker extraction regex pattern
SPEAKER_PATTERN = re.compile(r"^(?:Speaker\s*)?([A-Za-z]+(?:\s*Agent)?|MysteryShop(?:per)?):[\s]*", re.IGNORECASE)


def extract_speaker(text: str) -> tuple:
    """Extract speaker label from a line of text if present.
    
    Args:
        text: Line of text that may contain a speaker label
        
    Returns:
        Tuple of (speaker, cleaned_text)
    """
    # Clean up line first
    text = text.strip()
    
    # Try to extract speaker
    match = SPEAKER_PATTERN.match(text)
    if match:
        speaker = match.group(1).strip()
        cleaned_text = text[match.end():].strip()
        return (speaker, cleaned_text)
    
    return ("", text)


def split_by_speaker(text: str) -> list:
    """Split a multi-speaker chunk of text into list of (speaker, text) tuples.
    
    Args:
        text: Text chunk that may contain multiple speaker utterances
        
    Returns:
        List of (speaker, text) tuples
    """
    # Handle empty or None text
    if not text or pd.isna(text):
        return []
    
    # Split by lines but preserve paragraph structure
    lines = text.split('\n')
    
    segments = []
    current_speaker = ""
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if this line has a speaker label
        speaker, content = extract_speaker(line)
        
        if speaker:
            # If we were collecting content for a previous speaker, save it
            if current_speaker and current_content:
                full_content = " ".join(current_content).strip()
                if full_content:
                    segments.append((current_speaker, full_content))
                current_content = []
            
            # Start new speaker segment
            current_speaker = speaker
            if content:
                current_content.append(content)
        elif current_speaker:
            # Continue with current speaker if we have one
            current_content.append(line)
        else:
            # No speaker found but there's content
            segments.append(("", line))
    
    # Add the last segment if needed
    if current_speaker and current_content:
        full_content = " ".join(current_content).strip()
        if full_content:
            segments.append((current_speaker, full_content))
    
    return segments


def unchunk_groundtruth(input_path: str, output_path: str) -> None:
    """Read ground truth Excel file, unchunk it, and save as a new Excel file.
    
    Args:
        input_path: Path to input Excel file
        output_path: Path to output Excel file
    """
    print(f"Reading ground truth from {input_path}")
    
    try:
        # Read input file
        df = pd.read_excel(input_path, engine="openpyxl")
        
        if "Combined Text" not in df.columns:
            print("ERROR: Expected column 'Combined Text' not found in ground truth file")
            return
        
        # Process each row and split into speaker segments
        unchunked_segments = []
        
        for idx, row in df.iterrows():
            # Get the chunk index for reference
            chunk_index = row["Chunk Index"] if "Chunk Index" in df.columns else idx
            
            # Get combined text and split it
            combined_text = str(row["Combined Text"])
            segments = split_by_speaker(combined_text)
            
            # Add each segment to the result list
            for speaker, text in segments:
                if text.strip():  # Only add non-empty text
                    unchunked_segments.append({
                        "Original Chunk": chunk_index,
                        "Speaker": speaker,
                        "Text": text
                    })
        
        # Create DataFrame from unchunked segments
        result_df = pd.DataFrame(unchunked_segments)
        
        # Add an index column
        result_df.insert(0, "Segment Index", range(1, len(result_df) + 1))
        
        # Save to Excel
        print(f"Writing {len(result_df)} segments to {output_path}")
        result_df.to_excel(output_path, index=False)
        
        print(f"Success! {len(df)} chunks expanded to {len(result_df)} individual speaker segments")
    
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Get file paths from command line if provided
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    
    # Ensure file paths are Path objects
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    print("\n=== Ground Truth Unchunker ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not input_file.exists():
        sys.exit(f"ERROR: Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    unchunk_groundtruth(input_file, output_file)


if __name__ == "__main__":
    main()
