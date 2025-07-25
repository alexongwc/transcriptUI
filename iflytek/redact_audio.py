#!/usr/bin/env python3
"""
Audio Redaction Script for iFlytek Transcription
Redacts (silences) a specific portion of an audio file
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def time_to_seconds(time_str):
    """Convert time string (MM:SS or HH:MM:SS) to seconds"""
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError("Time format must be MM:SS or HH:MM:SS")

def format_ms(milliseconds):
    """Convert milliseconds to a readable time format HH:MM:SS"""
    seconds = milliseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def redact_audio_with_ffmpeg(input_file, output_file, start_time, end_time):
    """
    Redact (silence) a portion of an audio file using FFmpeg.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to save the redacted audio
        start_time: Start time of redaction in format "MM:SS" or "HH:MM:SS"
        end_time: End time of redaction in format "MM:SS" or "HH:MM:SS"
    """
    try:
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)

        if start_sec >= end_sec:
            print(f"❌ Error: Start time must be before end time")
            return False

        print(f"🎬 Redacting audio from {start_time} to {end_time} using FFmpeg...")

        # Construct the ffmpeg command
        # The 'volume' audio filter is used to set the volume to 0 (mute) between the specified timestamps.
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-af', f"volume=enable='between(t,{start_sec},{end_sec})':volume=0",
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error:\n{result.stderr}")
            return False
            
        print(f"✅ Successfully created redacted audio file: {output_file}")
        return True

    except FileNotFoundError:
        print("❌ Error: 'ffmpeg' command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Redact (silence) a portion of an audio file')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input_file_redacted.extension)')
    parser.add_argument('-s', '--start', help='Start time of redaction in format MM:SS or HH:MM:SS', required=True)
    parser.add_argument('-e', '--end', help='End time of redaction in format MM:SS or HH:MM:SS', required=True)
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        input_path = Path(input_file)
        output_file = str(input_path.with_name(f"{input_path.stem}_redacted{input_path.suffix}"))
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Redact audio using FFmpeg
    success = redact_audio_with_ffmpeg(input_file, output_file, args.start, args.end)
    
    if success:
        print(f"\n🎉 Redaction complete! Redacted audio saved to '{output_file}'.")
    else:
        print("\n❌ Redaction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
