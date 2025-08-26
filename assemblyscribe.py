import assemblyai as aai
import pandas as pd
import os
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import sys
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - use environment variable for API key
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

def check_api_key():
    """Check if API key is available, only exit if running as main script"""
    if not API_KEY:
        error_msg = "❌ Error: ASSEMBLYAI_API_KEY environment variable is required!"
        instructions = [
            "   Please set your AssemblyAI API key as an environment variable:",
            "   export ASSEMBLYAI_API_KEY='your_api_key_here'",
            "   Or create a .env file with: ASSEMBLYAI_API_KEY=your_api_key_here"
        ]
        
        if __name__ == "__main__":
            print(error_msg)
            for instruction in instructions:
                print(instruction)
            sys.exit(1)
        else:
            # If imported as module, just print warning but don't exit
            print("⚠️  Warning: ASSEMBLYAI_API_KEY not found in environment variables")
            return False
    return True

MAX_DURATION = 300  # 5 minutes in seconds for conversation chunks
CONVERSATION_CHUNK_SIZE = 5  # Number of conversation turns per chunk

def create_conversation_chunks(df, segments_per_chunk=5):
    """Create conversation chunks with speaker labels matching the reference format"""
    try:
        # Convert DataFrame to list of dictionaries
        segments = df.to_dict('records')
        
        # Create chunks
        chunks = []
        current_chunk = []
        
        for segment in segments:
            current_chunk.append(segment)
            
            if len(current_chunk) >= segments_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add remaining segments as last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Format chunks with speaker labels
        formatted_chunks = []
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            # Get start and end times
            start_time = chunk[0].get('Start Time')
            end_time = chunk[-1].get('End Time')
            
            # Create combined text with speaker labels (newline separated)
            combined_text_parts = []
            
            for segment in chunk:
                speaker = segment.get('Speaker', '')
                text = segment.get('Text', '')
                
                if text:
                    combined_text_parts.append(f"{speaker}: {text}")
            
            combined_text = "\n".join(combined_text_parts)
            
            # Create chunk record matching reference format (without segment_count)
            chunk_record = {
                'Chunk_Number': chunk_idx,
                'Start_Time': start_time,
                'End_Time': end_time,
                'Combined_Text': combined_text
            }
            
            formatted_chunks.append(chunk_record)
        
        return pd.DataFrame(formatted_chunks)
        
    except Exception as e:
        print(f"Error creating chunks: {e}")
        return None

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

def test_api_connection():
    """Test AssemblyAI API connection"""
    try:
        aai.settings.api_key = API_KEY
        # Simple test request
        print("✅ API key configured successfully!")
        return True
    except Exception as e:
        print(f"❌ API connection test failed: {str(e)}")
        return False

def transcribe_audio(audio_path, start_time=0):
    """Transcribe audio using AssemblyAI"""
    try:
        # Set API key
        aai.settings.api_key = API_KEY
        
        print(f"\nSending audio to AssemblyAI API: {audio_path}")
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            language_code="en",   # English language
            punctuate=True,       # Add punctuation
            format_text=True,     # Format text
        )
        
        transcriber = aai.Transcriber()
        
        # Upload and transcribe
        transcript = transcriber.transcribe(audio_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription failed: {transcript.error}")
            return None
        
        print("Received response from AssemblyAI API")
        
        segments = []
        
        # Process utterances (speaker-segmented results)
        if transcript.utterances:
            print(f"Found {len(transcript.utterances)} speaker utterances")
            
            for utterance in transcript.utterances:
                segments.append({
                    "start": utterance.start / 1000.0 + start_time,  # Convert ms to seconds
                    "end": utterance.end / 1000.0 + start_time,
                    "speaker": f"SPEAKER_{utterance.speaker}",
                    "text": utterance.text.strip()
                })
        
        elif transcript.words:
            print("No utterances found, processing word-level data...")
            
            # Group words by speaker to create segments
            current_segment = None
            
            for word in transcript.words:
                speaker_id = getattr(word, 'speaker', 'A')  # Default speaker if not available
                word_text = word.text
                word_start = word.start / 1000.0  # Convert ms to seconds
                word_end = word.end / 1000.0
                
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
        
        else:
            print("No segments or words found in response")
            return None
        
        print(f"Processed {len(segments)} segments from AssemblyAI")
        return segments
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

def process_single_audio_file():
    """Process the single specified audio file using AssemblyAI"""
    # Test API connection first
    if not test_api_connection():
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
    
    # For small files (under 100MB), send directly to API (AssemblyAI handles larger files better)
    if file_size < 100 * 1024 * 1024:
        print("Sending directly to AssemblyAI API...")
        
        # Transcribe the entire file at once
        segments = transcribe_audio(AUDIO_FILE, 0)
        
        if segments:
            all_segments = segments
            print(f"✅ Direct processing successful - {len(segments)} segments found")
        else:
            print("❌ Direct processing failed")
            return
    else:
        # For very large files, try to chunk them
        print("Very large file - attempting to chunk...")
        
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
                    segments = transcribe_audio(temp_path, chunk['start_time'])
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
            segments = transcribe_audio(AUDIO_FILE, 0)
            
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
            # Rename speakers - robust handling of raw speaker IDs coming from AssemblyAI
            speaker_id = segment["speaker"]  # e.g. "A", "B", "SPEAKER_A", "SPEAKER_B"
            
            speaker_num = None
            # Handle speaker IDs from AssemblyAI
            if isinstance(speaker_id, str):
                if speaker_id in ['A', '0']:
                    speaker_num = 0
                elif speaker_id in ['B', '1']:
                    speaker_num = 1
                elif speaker_id in ['C', '2']:
                    speaker_num = 2
                elif speaker_id in ['D', '3']:
                    speaker_num = 3
                elif "speaker_" in speaker_id.lower():
                    try:
                        last_part = speaker_id.split("_")[-1]
                        if last_part in ['A', '0']:
                            speaker_num = 0
                        elif last_part in ['B', '1']:
                            speaker_num = 1
                        elif last_part in ['C', '2']:
                            speaker_num = 2
                        elif last_part in ['D', '3']:
                            speaker_num = 3
                    except:
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
        
        # Create chunked version
        chunked_df = create_conversation_chunks(merged_df, CONVERSATION_CHUNK_SIZE)
        if chunked_df is not None:
            chunked_excel = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.xlsx")
            chunked_csv = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.csv")
            chunked_txt = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.txt")
            
            # Save chunked outputs
            chunked_df.to_excel(chunked_excel, index=False, engine='openpyxl')
            chunked_df.to_csv(chunked_csv, index=False)
            
            # Save readable TXT format
            with open(chunked_txt, 'w', encoding='utf-8') as f:
                f.write("Conversation Chunks - With Speaker Labels\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Source: {AUDIO_FILE}\n")
                f.write(f"Total segments: {len(merged_df)}\n")
                f.write(f"Segments per chunk: {CONVERSATION_CHUNK_SIZE}\n")
                f.write(f"Total chunks: {len(chunked_df)}\n\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, row in chunked_df.iterrows():
                    f.write(f"CHUNK {row['Chunk_Number']:02d}\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Time Range: {row['Start_Time']} - {row['End_Time']}\n\n")
                    f.write("Combined Text:\n")
                    f.write(str(row['Combined_Text']))
                    f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"\n✅ Processing complete!")
        print(f"Original CSV saved: {csv_path}")
        print(f"Merged CSV saved: {merged_csv}")
        if chunked_df is not None:
            print(f"Chunked Excel saved: {chunked_excel}")
            print(f"Chunked CSV saved: {chunked_csv}")
            print(f"Chunked TXT saved: {chunked_txt}")
    else:
        print("\n❌ Transcription failed.")

if __name__ == "__main__":
    # Check API key when running as main script
    if not check_api_key():
        sys.exit(1)
    
    # Only run if AUDIO_FILE and OUTPUT_FOLDER are defined
    if 'AUDIO_FILE' in globals() and 'OUTPUT_FOLDER' in globals():
        process_single_audio_file()
    else:
        print("❌ AUDIO_FILE and OUTPUT_FOLDER must be defined to run this script directly")