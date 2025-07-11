import requests
import pandas as pd
import os
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import sys
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
from urllib3.util.retry import Retry
from datetime import timedelta
import certifi
import ssl
import glob
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - use environment variable for API key
API_KEY = os.getenv("XI_API_KEY")

def check_api_key():
    """Check if API key is available, only exit if running as main script"""
    if not API_KEY:
        error_msg = "❌ Error: XI_API_KEY environment variable is required!"
        instructions = [
            "   Please set your ElevenLabs API key as an environment variable:",
            "   export XI_API_KEY='your_api_key_here'",
            "   Or create a .env file with: XI_API_KEY=your_api_key_here"
        ]
        
        if __name__ == "__main__":
            print(error_msg)
            for instruction in instructions:
                print(instruction)
            sys.exit(1)
        else:
            # If imported as module, just print warning but don't exit
            print("⚠️  Warning: XI_API_KEY not found in environment variables")
            return False
    return True

# API key will be checked when running as main script or when needed

MAX_DURATION = 300  # 5 minutes in seconds for conversation chunks
CONVERSATION_CHUNK_SIZE = 5  # Number of conversation turns per chunk

class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ciphers=None)
        kwargs['ssl_context'] = context
        return super(CustomHTTPAdapter, self).init_poolmanager(*args, **kwargs)

def format_timestamp(seconds):
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_ = total_seconds % 60
    milliseconds = int((td.microseconds / 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d},{milliseconds:03d}"

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def detect_conversation_chunks(audio_data, sample_rate, duration):
    """
    Create 5-minute audio chunks for better transcription quality.
    """
    chunks = []
    
    # If audio is short, process as single chunk
    if duration <= MAX_DURATION:
        chunks.append({
            'start_time': 0,
            'end_time': duration,
            'start_sample': 0,
            'end_sample': len(audio_data),
            'audio_data': audio_data
        })
        return chunks
    
    # Create 5-minute chunks
    chunk_duration = MAX_DURATION  # 5 minutes
    current_time = 0
    chunk_num = 0
    
    while current_time < duration:
        start_time = current_time
        end_time = min(duration, current_time + chunk_duration)
        
        # Convert to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't create empty chunks
        if end_sample > start_sample:
            chunks.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'audio_data': audio_data[start_sample:end_sample],
                'chunk_id': chunk_num + 1
            })
        
        current_time += chunk_duration
        chunk_num += 1
        
        # Break if we've covered the entire audio
        if end_time >= duration:
            break
    
    return chunks

def merge_overlapping_segments(all_segments):
    """
    Merge overlapping segments from different chunks to avoid duplicates.
    """
    if not all_segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(all_segments, key=lambda x: x["start"])
    merged_segments = []
    
    for segment in sorted_segments:
        # Check if this segment overlaps with the last merged segment
        if merged_segments and segment["start"] < merged_segments[-1]["end"]:
            # Check if it's a duplicate (same speaker, overlapping time)
            last_segment = merged_segments[-1]
            overlap_ratio = (min(segment["end"], last_segment["end"]) - max(segment["start"], last_segment["start"])) / (segment["end"] - segment["start"])
            
            # If significant overlap (>50%) and same speaker, skip this segment
            if overlap_ratio > 0.5 and segment["speaker"] == last_segment["speaker"]:
                continue
            # If different speakers but high overlap, keep the longer one
            elif overlap_ratio > 0.7:
                if (segment["end"] - segment["start"]) > (last_segment["end"] - last_segment["start"]):
                    merged_segments[-1] = segment
                continue
        
        merged_segments.append(segment)
    
    return merged_segments

def setup_requests_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
    adapter = CustomHTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.verify = certifi.where()
    return session

def test_api_connection():
    session = setup_requests_session()
    url = "https://api.elevenlabs.io/v1/models"
    
    try:
        response = session.get(url, headers={"xi-api-key": API_KEY})
        response.raise_for_status()
        print("✅ API connection test successful!")
        return session
    except Exception as e:
        print(f"❌ API connection test failed: {str(e)}")
        print("\nTrying alternative SSL configuration...")
        try:
            alt_session = requests.Session()
            alt_session.verify = False
            response = alt_session.get(url, headers={"xi-api-key": API_KEY})
            response.raise_for_status()
            print("✅ API connection successful with alternative SSL configuration!")
            return alt_session
        except Exception as e:
            print(f"❌ Alternative configuration also failed: {str(e)}")
            return None

def transcribe_audio(audio_path, session, start_time=0):
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": API_KEY}
    
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {"file": audio_file}
            data = {
                "model_id": "scribe_v1",
                "language_code": "en",
                "diarize": True,
                "tag_audio_events": True,
                "timestamps_granularity": "word",
                "output_format": "json",
                "enable_logging": False  # Zero Retention Mode – do not retain request/response data
            }
            
            print("\nSending audio to ElevenLabs API...")
            response = session.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            print("Received response from API")
            result = response.json()
            
            if "detail" in result:
                print(f"API Error: {result['detail']}")
                return None
            
            segments = []
            
            # Handle the new API response format
            if "words" in result:
                print("Found words in response, processing...")
                
                # Group words by speaker and create segments
                current_segment = None
                
                for word_data in result["words"]:
                    if word_data.get("type") == "word":  # Only process actual words, not spacing
                        speaker_id = word_data.get("speaker_id", "unknown")
                        word_text = word_data.get("text", "")
                        word_start = word_data.get("start", 0)
                        word_end = word_data.get("end", 0)
                        
                        # If this is a new speaker or first word, start a new segment
                        if current_segment is None or current_segment["speaker_id"] != speaker_id:
                            # Save previous segment if exists
                            if current_segment is not None:
                                segments.append({
                                    "start": current_segment["start"] + start_time,
                                    "end": current_segment["end"] + start_time,
                                    "speaker": f"SPEAKER_{current_segment['speaker_id']}",
                                    "text": current_segment["text"].strip()
                                })
                            
                            # Start new segment
                            current_segment = {
                                "speaker_id": speaker_id,
                                "start": word_start,
                                "end": word_end,
                                "text": word_text
                            }
                        else:
                            # Continue current segment
                            current_segment["end"] = word_end
                            current_segment["text"] += " " + word_text
                
                # Don't forget the last segment
                if current_segment is not None:
                    segments.append({
                        "start": current_segment["start"] + start_time,
                        "end": current_segment["end"] + start_time,
                        "speaker": f"SPEAKER_{current_segment['speaker_id']}",
                        "text": current_segment["text"].strip()
                    })
                
                print(f"Processed {len(segments)} segments from word-level data")
                
            elif "segments" in result:
                print("Found segments in response")
                for segment in result["segments"]:
                    # Adjust timestamps by start_time
                    segments.append({
                        "start": segment.get("start", 0) + start_time,
                        "end": segment.get("end", 0) + start_time,
                        "speaker": f"SPEAKER_{segment.get('speaker_id', '00')}",
                        "text": segment.get("text", "").strip()
                    })
                print(f"Processed {len(segments)} segments")
            else:
                print("No segments or words found in response")
                print("API Response keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
                return None
            
            return segments
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        return None

def process_single_audio_file():
    """Process the single specified audio file"""
    # Test API connection first
    session = test_api_connection()
    if session is None:
        print("Cannot proceed due to API connection issues.")
        return
    
    # Check if the audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        return
    
    # Create output directory if it doesn't exist
    create_folder_if_not_exists(OUTPUT_FOLDER)
    
    print(f"\nProcessing audio file: {AUDIO_FILE}")
    
    # Get file size and extension
    file_size = os.path.getsize(AUDIO_FILE)
    file_extension = os.path.splitext(AUDIO_FILE)[1].lower()
    
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"File extension: {file_extension}")
    
    # For MP3 files, use ffmpeg-based chunking for better quality
    if file_extension == '.mp3':
        print("MP3 file detected - using ffmpeg-based chunking for optimal quality...")
        all_segments = process_mp3_with_ffmpeg_chunking(session)
    # For small files (under 10MB), send directly to API
    elif file_size < 10 * 1024 * 1024:
        print("Small file - sending directly to ElevenLabs API...")
        
        # Transcribe the entire file at once
        segments = transcribe_audio(AUDIO_FILE, session, 0)
        
        if segments:
            all_segments = segments
            print(f"✅ Direct processing successful - {len(segments)} segments found")
        else:
            print("❌ Direct processing failed")
            return
    else:
        # For larger WAV files, try to chunk them
        print("Large file - attempting to chunk...")
        
        try:
            # Try to load audio file with soundfile
            audio_data, sample_rate = sf.read(AUDIO_FILE)
            duration = len(audio_data) / sample_rate
            
            # Create temp directory for conversation chunks
            temp_dir = "temp_segments"
            create_folder_if_not_exists(temp_dir)
            
            try:
                all_segments = []
                
                # Create 5-minute chunks
                print(f"Audio duration: {duration:.2f} seconds, creating 5-minute chunks")
                chunks = detect_conversation_chunks(audio_data, sample_rate, duration)
                
                print(f"Created {len(chunks)} five-minute chunks for processing")
                
                for i, chunk in enumerate(chunks):
                    # Save chunk temporarily
                    temp_path = os.path.join(temp_dir, f"chunk_{i+1}.wav")
                    sf.write(temp_path, chunk['audio_data'], sample_rate)
                    
                    print(f"\nProcessing chunk {i+1}/{len(chunks)} ({format_timestamp(chunk['start_time'])} - {format_timestamp(chunk['end_time'])})")
                    
                    # Transcribe chunk
                    segments = transcribe_audio(temp_path, session, chunk['start_time'])
                    if segments:
                        all_segments.extend(segments)
                        print(f"✅ Chunk {i+1} processed successfully - {len(segments)} segments found")
                    else:
                        print(f"❌ Failed to process chunk {i+1}")
                    
                    # Clean up temporary file
                    os.remove(temp_path)
            
            finally:
                # Clean up temp directory
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error loading audio file for chunking: {str(e)}")
            print("Falling back to direct API processing...")
            
            # Fallback: send directly to API
            segments = transcribe_audio(AUDIO_FILE, session, 0)
            
            if segments:
                all_segments = segments
                print(f"✅ Fallback processing successful - {len(segments)} segments found")
            else:
                print("❌ Fallback processing also failed")
                return
    
    if all_segments:
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Convert to dataframe format
        df_segments = []
        for segment in all_segments:
            # Rename speakers - robust handling of raw speaker IDs coming from ElevenLabs
            speaker_id = segment["speaker"]  # e.g. "0", "1", "speaker_0", "SPEAKER_1", "SPEAKER_speaker_2"

            speaker_num = None
            # Handle integer IDs directly
            if isinstance(speaker_id, (int, float)) and str(speaker_id).isdigit():
                speaker_num = int(speaker_id)
            elif isinstance(speaker_id, str):
                if speaker_id.isdigit():
                    speaker_num = int(speaker_id)
                elif "speaker_" in speaker_id.lower():
                    try:
                        speaker_num = int(speaker_id.split("_")[-1])
                    except ValueError:
                        speaker_num = None
                elif speaker_id.upper().startswith("SPEAKER_"):
                    try:
                        speaker_num = int(speaker_id.split("_")[-1])
                    except ValueError:
                        speaker_num = None

            if speaker_num is not None:
                speaker_name = "Mysteryshopper" if speaker_num % 2 == 0 else "InsuranceAgent"
            else:
                # Fallback – keep the raw ID if we cannot parse a number
                speaker_name = speaker_id

            df_segments.append({
                "Start Time": format_timestamp(segment["start"]),
                "End Time": format_timestamp(segment["end"]),
                "Speaker": speaker_name,
                "Text": segment["text"],
                "Label": ""
            })
        
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
        
        # Save full transcript to CSV in output folder
        df = pd.DataFrame(df_segments)
        csv_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.csv")
        df.to_csv(csv_path, index=False)
        
        # Create merged version by combining consecutive segments from same speaker
        merged_rows = []
        prev = None
        
        for segment in df_segments:
            if prev is None:
                prev = segment.copy()
            elif segment['Speaker'] == prev['Speaker']:
                # Merge with previous segment from same speaker
                prev['End Time'] = segment['End Time']
                prev['Text'] = str(prev['Text']) + ' ' + str(segment['Text'])
            else:
                merged_rows.append(prev)
                prev = segment.copy()
        
        if prev is not None:
            merged_rows.append(prev)
        
        # Save merged version
        merged_df = pd.DataFrame(merged_rows)
        merged_df = merged_df[["Start Time", "End Time", "Speaker", "Text", "Label"]]
        merged_csv = os.path.join(OUTPUT_FOLDER, f"{base_name}_merged.csv")
        merged_df.to_csv(merged_csv, index=False)
        
        print(f"\n✅ Processing complete!")
        print(f"Original CSV saved: {csv_path}")
        print(f"Merged CSV saved: {merged_csv}")
    else:
        print("\n❌ Transcription failed.")

def process_mp3_with_ffmpeg_chunking(session):
    """Process MP3 files using ffmpeg for chunking to maintain quality"""
    import subprocess
    
    try:
        # Get audio duration using ffprobe
        print("Analyzing MP3 file duration...")
        cmd = [
            'ffprobe', '-i', AUDIO_FILE, '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Could not analyze MP3 file. Falling back to direct processing.")
            return transcribe_audio(AUDIO_FILE, session, 0)
            
        duration = float(result.stdout.strip())
        chunk_duration = MAX_DURATION  # 5 minutes
        
        # Calculate number of chunks
        num_chunks = int(duration / chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
        
        if num_chunks == 1:
            # File is short, process directly
            print("MP3 file is short, processing directly...")
            return transcribe_audio(AUDIO_FILE, session, 0)
        
        print(f"Creating {num_chunks} five-minute chunks from MP3 for optimal quality")
        
        # Create temp directory for chunks
        temp_dir = "temp_mp3_segments"
        create_folder_if_not_exists(temp_dir)
        
        all_segments = []
        
        try:
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, duration)
                
                print(f"\nProcessing MP3 chunk {i+1}/{num_chunks} ({format_timestamp(start_time)} - {format_timestamp(end_time)})")
                
                # Create chunk using ffmpeg (convert to WAV for better API compatibility)
                chunk_path = os.path.join(temp_dir, f"chunk_{i+1}.wav")
                cmd = [
                    'ffmpeg', '-i', AUDIO_FILE, '-ss', str(start_time),
                    '-t', str(end_time - start_time), '-ar', '16000', '-ac', '1', 
                    '-c:a', 'pcm_s16le', chunk_path, '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    print(f"❌ Failed to create MP3 chunk {i+1}")
                    continue
                
                # Check if chunk file was created and has content
                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                    print(f"❌ MP3 chunk {i+1} is empty or not created")
                    continue
                
                chunk_size = os.path.getsize(chunk_path)
                print(f"   📁 MP3 chunk {i+1} created: {chunk_size / 1024:.1f} KB")
                
                # Transcribe chunk
                segments = transcribe_audio(chunk_path, session, start_time)
                if segments:
                    all_segments.extend(segments)
                    print(f"✅ MP3 chunk {i+1} processed - {len(segments)} segments found")
                else:
                    print(f"❌ Failed to process MP3 chunk {i+1}")
                
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        finally:
            # Clean up temp directory
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        return all_segments
        
    except Exception as e:
        print(f"❌ Error during MP3 chunking: {str(e)}")
        print("Falling back to direct MP3 processing...")
        return transcribe_audio(AUDIO_FILE, session, 0)

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Check API key when running as main script
    if not check_api_key():
        sys.exit(1)
    
    # Only run if AUDIO_FILE and OUTPUT_FOLDER are defined
    if 'AUDIO_FILE' in globals() and 'OUTPUT_FOLDER' in globals():
        process_single_audio_file()
    else:
        print("❌ AUDIO_FILE and OUTPUT_FOLDER must be defined to run this script directly")
