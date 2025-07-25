#!/usr/bin/env python3
"""
Audio Conversion Script for iFlytek Transcription
Converts audio files to the required format: 16kHz, 16-bit, mono WAV
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def convert_with_ffmpeg(input_file, output_file):
    """Convert audio file using FFmpeg"""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-ar', '16000',  # Sample rate: 16kHz
        '-ac', '1',      # Channels: mono
        '-sample_fmt', 's16',  # 16-bit signed integer
        '-y',            # Overwrite output file
        output_file
    ]
    
    try:
        print(f"🔄 Converting {input_file} to {output_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to {output_file}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running FFmpeg: {e}")
        return False

def convert_with_librosa(input_file, output_file):
    """Convert audio file using librosa (fallback method)"""
    try:
        import librosa
        import soundfile as sf
        
        print(f"🔄 Converting {input_file} to {output_file} using librosa...")
        
        # Load audio file
        audio, sr = librosa.load(input_file, sr=16000, mono=True)
        
        # Save as WAV
        sf.write(output_file, audio, 16000, subtype='PCM_16')
        
        print(f"✅ Successfully converted to {output_file}")
        return True
        
    except ImportError:
        print("❌ librosa and soundfile not installed. Please install with:")
        print("pip install librosa soundfile")
        return False
    except Exception as e:
        print(f"❌ Error with librosa conversion: {e}")
        return False

def get_audio_info(file_path):
    """Get audio file information using FFprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    return {
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0)),
                        'duration': float(stream.get('duration', 0)),
                        'codec': stream.get('codec_name', 'unknown')
                    }
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not get audio info: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert audio files for iFlytek transcription')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path (default: zhongwen.wav)')
    parser.add_argument('--info', action='store_true', help='Show audio file information only')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output or 'zhongwen.wav'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Show file info if requested
    if args.info:
        info = get_audio_info(input_file)
        if info:
            print(f"📊 Audio file information for {input_file}:")
            print(f"  Sample Rate: {info['sample_rate']} Hz")
            print(f"  Channels: {info['channels']}")
            print(f"  Duration: {info['duration']:.2f} seconds")
            print(f"  Codec: {info['codec']}")
            
            # Check if conversion is needed
            if (info['sample_rate'] == 16000 and 
                info['channels'] == 1 and 
                input_file.lower().endswith('.wav')):
                print("✅ File is already in the correct format!")
            else:
                print("🔄 File needs conversion")
        sys.exit(0)
    
    # Check conversion tools
    if check_ffmpeg():
        success = convert_with_ffmpeg(input_file, output_file)
    else:
        print("⚠️  FFmpeg not found, trying librosa...")
        success = convert_with_librosa(input_file, output_file)
    
    if success:
        # Verify output file
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📁 Output file size: {file_size:,} bytes")
            
            # Show info about converted file
            info = get_audio_info(output_file)
            if info:
                print(f"📊 Converted file info:")
                print(f"  Sample Rate: {info['sample_rate']} Hz")
                print(f"  Channels: {info['channels']}")
                print(f"  Duration: {info['duration']:.2f} seconds")
        
        print(f"\n🎉 Conversion complete! You can now use '{output_file}' for transcription.")
        print("Run the transcription script with:")
        print("python istdemo_improved.py")
        
    else:
        print("\n❌ Conversion failed!")
        print("\nManual conversion options:")
        print("1. Install FFmpeg: https://ffmpeg.org/download.html")
        print("2. Install Python libraries: pip install librosa soundfile")
        print("3. Use online converter to convert to 16kHz mono WAV")
        sys.exit(1)

if __name__ == "__main__":
    main() 