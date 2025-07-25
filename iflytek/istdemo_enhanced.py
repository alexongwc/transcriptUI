import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime, timedelta
from time import mktime
import _thread as thread
import os
import wave
import struct

# Configuration based on API documentation
CONFIG = {
    'wsUrl': "ws://94.74.125.108:9990/tuling/ast/v3",
    'appId': "123456",  # Replace with your actual App ID
    'bizId': "39769795890",  # Replace with your actual Business ID
    'traceId': "traceId123456",  # Unique trace ID for this session
    'frameSize': 1280,  # Audio frame size in bytes
    'interval': 0.04,  # Frame sending interval in seconds
    'required_sample_rate': 16000,  # Required sample rate
    'required_channels': 1,  # Required number of channels
    'required_sample_width': 2,  # Required sample width (16-bit = 2 bytes)
}

# Frame status constants
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

# Global variables for enhanced tracking
transcription_segments = []
current_segment = ""
transcription_complete = False
audio_start_time = None
current_speaker = "speaker_0"  # Default speaker
frame_count = 0
audio_duration_per_frame = CONFIG['frameSize'] / (CONFIG['required_sample_rate'] * CONFIG['required_sample_width'])

def validate_audio_file(file_path):
    """
    Validate audio file format and properties
    """
    if not os.path.exists(file_path):
        print(f"❌ Audio file not found: {file_path}")
        return False
    
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
            duration = frames / sample_rate
            
            print(f"Audio file info:")
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample Width: {sample_width} bytes ({sample_width * 8}-bit)")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Total Frames: {frames}")
            
            if (sample_rate == CONFIG['required_sample_rate'] and 
                channels == CONFIG['required_channels'] and 
                sample_width == CONFIG['required_sample_width']):
                print("✅ Audio format is compatible!")
                return True
            else:
                print("❌ Audio format is not compatible!")
                print(f"Required: {CONFIG['required_sample_rate']}Hz, {CONFIG['required_channels']} channel(s), {CONFIG['required_sample_width']} bytes")
                return False
                
    except Exception as e:
        print(f"❌ Error validating audio file: {e}")
        return False

def calculate_timestamp(frame_number):
    """
    Calculate relative timestamp based on frame number and audio properties
    Returns format: HH:MM:SS.mmm (starting from 00:00:00)
    """
    elapsed_seconds = frame_number * audio_duration_per_frame
    
    # Convert to hours, minutes, seconds, milliseconds
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    milliseconds = int((elapsed_seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def identify_speaker(text_content, segment_index):
    """
    Identify speaker as either mysteryshopper or insuranceagent
    According to memory: always identify the first speaker (speaker_0) as the mysteryshopper
    """
    # First speaker is always mysteryshopper
    if segment_index == 0:
        return "mysteryshopper"
    
    # Mystery shopper indicators (questions, greetings, inquiries)
    mysteryshopper_keywords = [
        "hello", "hi", "good morning", "good afternoon", "good evening",
        "can you", "could you", "would you", "do you", "are you",
        "what", "when", "where", "why", "how", "which",
        "i'm looking for", "i need", "i want", "i'm interested",
        "can i", "may i", "is it possible", "tell me about",
        "what about", "how about", "pricing", "cost", "price",
        "thank you", "thanks", "okay", "alright", "sounds good"
    ]
    
    # Insurance agent indicators (responses, explanations, offers)
    insuranceagent_keywords = [
        "yes", "no", "sure", "certainly", "of course", "absolutely",
        "let me", "i can", "we offer", "we provide", "we have",
        "our policy", "our coverage", "our plan", "our service",
        "premium", "deductible", "coverage", "benefit", "claim",
        "insurance", "policy", "quote", "rate", "discount",
        "you're welcome", "happy to help", "glad to assist"
    ]
    
    text_lower = text_content.lower()
    
    # Check for mysteryshopper keywords
    if any(keyword in text_lower for keyword in mysteryshopper_keywords):
        return "mysteryshopper"
    
    # Check for insuranceagent keywords
    if any(keyword in text_lower for keyword in insuranceagent_keywords):
        return "insuranceagent"
    
    # Default alternating logic: odd segments = mysteryshopper, even = insuranceagent
    return "mysteryshopper" if segment_index % 2 == 1 else "insuranceagent"

def on_message(ws, message):
    """
    Handle incoming WebSocket messages with enhanced processing
    """
    global transcription_segments, current_segment, transcription_complete, current_speaker, frame_count
    
    try:
        data = json.loads(message)
        header = data.get("header", {})
        payload = data.get("payload", {})
        
        code = header.get("code", -1)
        sid = header.get("sid", "unknown")
        status = header.get("status", -1)
        
        timestamp = calculate_timestamp(frame_count)
        
        print(f"📨 [{timestamp}] Received message - Code: {code}, Status: {status}, SID: {sid}")
        
        if code != 0:
            print(f"❌ Error from server: {data}")
            return
        
        # Process transcription results
        if "result" in payload:
            result = payload["result"]
            pgs = result.get("pgs", "")
            ws_list = result.get("ws", [])
            
            if pgs == 'apd':  # Append mode
                segment_text = ""
                for ws_item in ws_list:
                    # Check if there's speaker information in the response
                    speaker_info = ws_item.get("speaker", None)
                    
                    for word_info in ws_item.get("cw", []):
                        word = word_info.get("w", "")
                        confidence = word_info.get("confidence", 0)
                        
                        # Extract timing information if available
                        word_start = word_info.get("wb", 0)  # Word begin time
                        word_end = word_info.get("we", 0)    # Word end time
                        
                        segment_text += word
                        current_segment += word
                
                if segment_text.strip():
                    # Identify speaker for this segment
                    speaker = identify_speaker(segment_text, len(transcription_segments))
                    
                    # Create detailed segment information
                    segment_info = {
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "text": segment_text.strip(),
                        "confidence": "high",  # Could be calculated from word confidences
                        "frame_number": frame_count,
                        "raw_data": ws_list  # Keep raw data for debugging
                    }
                    
                    transcription_segments.append(segment_info)
                    print(f"📝 [{timestamp}] {speaker}: {segment_text.strip()}")
        
        # Check if transcription is complete
        if status == 2:  # Final frame processed
            transcription_complete = True
            print("✅ Transcription completed!")
            ws.close()
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw message: {message}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")

def on_error(ws, error):
    """
    Handle WebSocket errors
    """
    print(f"❌ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Handle WebSocket connection close
    """
    print(f"🔌 Connection closed - Status: {close_status_code}, Message: {close_msg}")

def on_open(ws):
    """
    Handle WebSocket connection open and start audio streaming
    """
    global audio_start_time, frame_count
    
    def run(*args):
        global frame_count, audio_start_time
        
        audio_file = "zhongwen.wav"  # Change this to your audio file path
        
        print(f"🎵 Starting audio streaming from: {audio_file}")
        
        # Validate audio file first
        if not validate_audio_file(audio_file):
            print("❌ Audio file validation failed. Please convert your audio file first.")
            ws.close()
            return
        
        audio_start_time = datetime.now()
        
        try:
            with open(audio_file, "rb") as fp:
                frame_count = 0
                status = STATUS_FIRST_FRAME
                
                while True:
                    buf = fp.read(CONFIG['frameSize'])
                    
                    if not buf:
                        status = STATUS_LAST_FRAME
                    
                    # Prepare message payload
                    message_data = {
                        "header": {
                            "traceId": CONFIG['traceId'],
                            "appId": CONFIG['appId'],
                            "bizId": CONFIG['bizId'],
                            "status": status
                        },
                        "payload": {
                            "audio": {
                                "audio": base64.b64encode(buf).decode('utf-8') if buf else ""
                            }
                        }
                    }
                    
                    # Send message
                    ws.send(json.dumps(message_data))
                    frame_count += 1
                    
                    if frame_count % 25 == 0:
                        timestamp = calculate_timestamp(frame_count)
                        print(f"📤 [{timestamp}] Sent frame {frame_count} ({len(buf)} bytes)")
                    
                    if status == STATUS_LAST_FRAME:
                        print(f"📤 Sent final frame {frame_count}")
                        break
                    
                    # Update status for next iteration
                    if status == STATUS_FIRST_FRAME:
                        status = STATUS_CONTINUE_FRAME
                    
                    # Simulate audio sampling interval
                    time.sleep(CONFIG['interval'])
                    
        except Exception as e:
            print(f"❌ Error reading audio file: {e}")
            ws.close()

    thread.start_new_thread(run, ())

def save_transcription_results():
    """
    Save transcription results in multiple formats
    """
    if not transcription_segments:
        print("⚠️  No transcription segments to save")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory if it doesn't exist
    results_dir = "transcription_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed JSON format
    json_filename = f"{results_dir}/transcription_{timestamp}.json"
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_segments": len(transcription_segments),
                    "audio_file": "converted_audio.wav",
                    "processing_time": str(datetime.now() - audio_start_time) if audio_start_time else "unknown"
                },
                "segments": transcription_segments
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed results saved to: {json_filename}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
    
    # Save formatted text transcript
    txt_filename = f"{results_dir}/transcript_{timestamp}.txt"
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AUDIO TRANSCRIPTION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Segments: {len(transcription_segments)}\n")
            f.write(f"Audio File: {audio_file if 'audio_file' in locals() else 'unknown'}\n")
            f.write("=" * 80 + "\n\n")
            
            current_speaker = None
            for i, segment in enumerate(transcription_segments):
                if segment['speaker'] != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"[{segment['speaker'].upper()}]\n")
                    current_speaker = segment['speaker']
                
                f.write(f"[{segment['timestamp']}] {segment['text']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SPEAKER SUMMARY\n")
            f.write("=" * 80 + "\n")
            
            speakers = {}
            for segment in transcription_segments:
                speaker = segment['speaker']
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(segment['text'])
            
            for speaker, texts in speakers.items():
                f.write(f"\n{speaker.upper()}:\n")
                f.write(f"  Total segments: {len(texts)}\n")
                f.write(f"  Sample text: {texts[0][:100]}{'...' if len(texts[0]) > 100 else ''}\n")
        
        print(f"📄 Formatted transcript saved to: {txt_filename}")
    except Exception as e:
        print(f"❌ Error saving transcript: {e}")
    
    # Save CSV format for analysis
    csv_filename = f"{results_dir}/segments_{timestamp}.csv"
    try:
        import csv
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Speaker', 'Text', 'Confidence', 'Frame'])
            
            for segment in transcription_segments:
                writer.writerow([
                    segment['timestamp'],
                    segment['speaker'],
                    segment['text'],
                    segment['confidence'],
                    segment['frame_number']
                ])
        
        print(f"📊 CSV data saved to: {csv_filename}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

def main():
    """
    Main function to run the enhanced transcription
    """
    global transcription_segments, transcription_complete, audio_start_time
    
    print("🎙️  Enhanced iFlytek Audio Transcription Client")
    print("=" * 60)
    print("Features: Timestamps, Speaker Diarization, Multi-format Output")
    print("=" * 60)
    
    # Reset global variables
    transcription_segments = []
    transcription_complete = False
    audio_start_time = None
    
    start_time = datetime.now()
    
    try:
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            CONFIG['wsUrl'],
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.on_open = on_open
        
        print(f"🔗 Connecting to: {CONFIG['wsUrl']}")
        
        # Run WebSocket connection
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        # Wait for transcription to complete
        while not transcription_complete:
            time.sleep(0.1)
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("📋 TRANSCRIPTION COMPLETED")
        print("=" * 60)
        print(f"Total segments: {len(transcription_segments)}")
        print(f"Processing time: {processing_time}")
        
        # Display speaker summary
        speakers = {}
        for segment in transcription_segments:
            speaker = segment['speaker']
            if speaker not in speakers:
                speakers[speaker] = 0
            speakers[speaker] += 1
        
        print("\n👥 Speaker Summary:")
        for speaker, count in speakers.items():
            print(f"  {speaker}: {count} segments")
        
        print("=" * 60)
        
        # Save results in multiple formats
        save_transcription_results()
        
        # Display recent segments
        print("\n📝 Recent Transcription Segments:")
        for segment in transcription_segments[-5:]:  # Show last 5 segments
            print(f"  [{segment['timestamp']}] {segment['speaker']}: {segment['text']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Transcription interrupted by user")
        if transcription_segments:
            save_transcription_results()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if transcription_segments:
            save_transcription_results()

if __name__ == "__main__":
    main() 