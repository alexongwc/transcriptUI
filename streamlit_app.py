import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile
import io
import traceback
from datetime import datetime

# Global switch: turn conversation chunking on/off
ENABLE_CHUNKING = False  # Set to True to create _chunked.csv and other chunk files

# No config import needed for the reverted approach
# Load environment variables from .env file (for local development)
# For Streamlit Cloud, secrets are handled via st.secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available in Streamlit Cloud

# Initialize session state for logging
def init_logging():
    """Initialize logging containers in session state"""
    if 'process_logs' not in st.session_state:
        st.session_state.process_logs = []
    if 'error_logs' not in st.session_state:
        st.session_state.error_logs = []
    if 'log_container' not in st.session_state:
        st.session_state.log_container = None
    if 'tx_logs' not in st.session_state:
        st.session_state.tx_logs = ""
    if 'ck_logs' not in st.session_state:
        st.session_state.ck_logs = ""
    if 'gap_log_messages' not in st.session_state:
        st.session_state.gap_log_messages = []

def add_log(message, log_type="INFO"):
    """Add a log message to the session state"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{log_type}] {message}"
    
    # Add to appropriate log collection
    if not hasattr(st.session_state, 'process_logs'):
        st.session_state.process_logs = []
    if not hasattr(st.session_state, 'error_logs'):
        st.session_state.error_logs = []
        
    st.session_state.process_logs.append(log_entry)
    if log_type in ["ERROR", "WARNING"]:
        st.session_state.error_logs.append(log_entry)

def display_logs():
    """Display all logs from session state in the original format"""
    # Transcription Logs
    if hasattr(st.session_state, 'tx_logs') and st.session_state.tx_logs:
        with st.expander("🔍 Transcription Logs", expanded=True):
            # Parse and display the JSON response in a readable format
            try:
                # Try to extract the actual log content if it's in JSON format
                if isinstance(st.session_state.tx_logs, str) and st.session_state.tx_logs.strip().startswith('{'):
                    import json
                    log_data = json.loads(st.session_state.tx_logs)
                    if isinstance(log_data, dict):
                        for key, value in log_data.items():
                            st.write(f"{key}: {value}")
                else:
                    # Display as regular text if not JSON
                    st.text_area("Status", value=st.session_state.tx_logs, height=150, disabled=True, key="tx_logs_display")
            except:
                # Fallback to raw text display if JSON parsing fails
                st.text_area("Status", value=st.session_state.tx_logs, height=150, disabled=True, key="tx_logs_fallback")

    # Gap Detection Logs
    if hasattr(st.session_state, 'gap_log_messages') and st.session_state.gap_log_messages:
        with st.expander("⏱️ Gap Detection Logs", expanded=True):
            if isinstance(st.session_state.gap_log_messages, list):
                for message in st.session_state.gap_log_messages:
                    st.write(message)
            else:
                st.write(st.session_state.gap_log_messages)

    # Error Logs
    if hasattr(st.session_state, 'error_logs') and st.session_state.error_logs:
        with st.expander("⚠️ Error Logs", expanded=True):
            for error in st.session_state.error_logs:
                st.error(error)

    # Process Logs (if any additional logs exist)
    if hasattr(st.session_state, 'process_logs') and st.session_state.process_logs:
        with st.expander("📋 Process Logs", expanded=True):
            st.text_area(
                "Processing History",
                value="\n".join(st.session_state.process_logs),
                height=200,
                disabled=True,
                key="process_logs_display"
            )

def create_logs_excel(output_folder, base_name, process_logs, error_logs):
    """Create Excel file with separate tabs for process and error logs"""
    try:
        # Create DataFrames for logs
        process_df = pd.DataFrame({'Log_Entry': process_logs})
        error_df = pd.DataFrame({'Error_Log_Entry': error_logs})
        
        # Create Excel file with multiple sheets
        excel_path = os.path.join(output_folder, f"{base_name}_logs.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            process_df.to_excel(writer, sheet_name='Process_Logs', index=False)
            error_df.to_excel(writer, sheet_name='Error_Logs', index=False)
        
        return excel_path
    except Exception as e:
        add_log(f"Failed to create logs Excel: {str(e)}", "ERROR")
        return None

# Optional DOCX generation
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Define format_timestamp function locally to avoid circular import
def format_timestamp(seconds):
    """Format seconds to HH:MM:SS,mmm format"""
    from datetime import timedelta
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_ = total_seconds % 60
    milliseconds = int((td.microseconds / 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d},{milliseconds:03d}"

def get_api_key():
    """Get API key from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'XI_API_KEY' in st.secrets:
            return st.secrets['XI_API_KEY']
    except:
        pass
    
    # Fallback to environment variable (for local development)
    api_key = os.getenv("XI_API_KEY")
    if api_key:
        return api_key
    
    # No API key found
    return None

def run_elevenlabs_transcription(audio_file_path, output_folder):
    """Run the elevenlabscribe.py script to process audio"""
    try:
        add_log("Starting ElevenLabs transcription process")
        
        # Get the absolute path to the audioUI directory
        audioui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(audioui_dir, "elevenlabscribe.py")
        
        add_log(f"Using script path: {script_path}")
        
        # Create a temporary modified version of elevenlabscribe.py
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Replace the hardcoded paths with absolute paths
        # Build path strings
        abs_audio = os.path.abspath(audio_file_path)
        abs_out   = os.path.abspath(output_folder)

        add_log(f"Processing audio file: {abs_audio}")
        add_log(f"Output folder: {abs_out}")

        # Prepend explicit vars to ensure they exist even if originals are commented out
        preface = (
            f'AUDIO_FILE = "{abs_audio}"\n'
            f'OUTPUT_FOLDER = "{abs_out}"\n'
        )

        # Remove any existing hard-coded definitions (commented or not) to avoid confusion
        cleaned = []
        for line in script_content.splitlines():
            # Remove lines that assign AUDIO_FILE or OUTPUT_FOLDER
            if line.strip().startswith('AUDIO_FILE') or line.strip().startswith('OUTPUT_FOLDER'):
                continue
            cleaned.append(line)
        modified_content = preface + '\n'.join(cleaned)
        
        # Write temporary script
        temp_script = os.path.join(output_folder, "temp_elevenlabscribe.py")
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        add_log("Created temporary script, executing transcription...")
        
        # Create environment for subprocess with API key
        env = os.environ.copy()
        api_key = get_api_key()
        if api_key:
            env["XI_API_KEY"] = api_key
        
        # Run the script from the audioUI directory
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, cwd=audioui_dir, env=env)
        
        # Clean up temp script
        os.remove(temp_script)
        
        log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        
        if result.returncode == 0:
            add_log("✅ Transcription completed successfully!")
            st.success("✅ Transcription completed successfully!")
            return True, log_text
        else:
            add_log("❌ Transcription failed", "ERROR")
            add_log(f"Error details: {log_text}", "ERROR")
            st.error("❌ Transcription failed")
            return False, log_text
            
    except Exception as e:
        error_msg = f"❌ Error running transcription: {str(e)}"
        add_log(error_msg, "ERROR")
        add_log(f"Traceback: {traceback.format_exc()}", "ERROR")
        st.error(error_msg)
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, f"Exception: {str(e)}"

def run_chunking(csv_file_path, output_folder):
    """Run the chunk.py script to create conversation chunks"""
    try:
        add_log("Starting chunking process")
        
        # Get the absolute path to the audioUI directory
        audioui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(audioui_dir, "chunk.py")
        
        add_log(f"Using chunking script: {script_path}")
        add_log(f"Input CSV: {csv_file_path}")
        
        # Create a temporary modified version of chunk.py
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Replace the hardcoded paths with absolute paths
        modified_content = script_content.replace(
            'INPUT_CSV = "/home/alexong/intage/1476_JunKaiOng_GEFA_m2_merged.csv"',
            f'INPUT_CSV = "{os.path.abspath(csv_file_path)}"'
        ).replace(
            'OUTPUT_FOLDER = "/home/alexong/intage"',
            f'OUTPUT_FOLDER = "{os.path.abspath(output_folder)}"'
        )
        
        # Also replace any other hardcoded paths that might exist
        modified_content = modified_content.replace(
            '/home/alexong/intage',
            f'{os.path.abspath(output_folder)}'
        )
        
        # Write temporary script
        temp_script = os.path.join(output_folder, "temp_chunk.py")
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        add_log("Created temporary chunking script, executing...")
        
        # Run the script from the audioUI directory
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, cwd=audioui_dir)
        
        # Clean up temp script
        os.remove(temp_script)
        
        log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        if result.returncode == 0:
            add_log("✅ Chunking completed successfully!")
            st.success("✅ Chunking completed successfully!")
            return True, log_text
        else:
            add_log("❌ Chunking failed", "ERROR")
            add_log(f"Chunking error details: {log_text}", "ERROR")
            st.error("❌ Chunking failed")
            return False, log_text
            
    except Exception as e:
        error_msg = f"❌ Error running chunking: {str(e)}"
        add_log(error_msg, "ERROR")
        add_log(f"Chunking traceback: {traceback.format_exc()}", "ERROR")
        st.error(error_msg)
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, f"Exception: {str(e)}"

# Enhanced find_output_files: accept both _chunked and _merged_chunked naming
def find_output_files(output_folder, base_name):
    """Find the generated output files"""
    # The elevenlabscribe.py creates files like: base_name_merged.csv
    # The chunk.py creates files like: base_name_merged_chunked.csv
    files = {
        'merged_csv': os.path.join(output_folder, f"{base_name}_merged.csv"),
        'chunked_csv': os.path.join(output_folder, f"{base_name}_merged_chunked.csv"),
        'chunked_excel': os.path.join(output_folder, f"{base_name}_merged_chunked.xlsx"),
        'chunked_txt': os.path.join(output_folder, f"{base_name}_merged_chunked.txt")
    }
    # Fallback names
    fallback = {
        # same naming without "merged_" prefix
        'chunked_csv': os.path.join(output_folder, f"{base_name}_chunked.csv"),
        'chunked_excel': os.path.join(output_folder, f"{base_name}_chunked.xlsx"),
        'chunked_txt': os.path.join(output_folder, f"{base_name}_chunked.txt"),
        # full-file path variants (previous implementation)
        'chunked_csv_full': os.path.join(output_folder, f"{base_name}_full_chunked.csv"),
        'chunked_excel_full': os.path.join(output_folder, f"{base_name}_full_chunked.xlsx"),
        'chunked_txt_full': os.path.join(output_folder, f"{base_name}_full_chunked.txt"),
        # NEW: full-file *merged* variants
        'merged_csv_full': os.path.join(output_folder, f"{base_name}_full_merged.csv"),
        'chunked_csv_full_merged': os.path.join(output_folder, f"{base_name}_full_merged_chunked.csv"),
        'chunked_excel_full_merged': os.path.join(output_folder, f"{base_name}_full_merged_chunked.xlsx"),
        'chunked_txt_full_merged': os.path.join(output_folder, f"{base_name}_full_merged_chunked.txt")
    }
    
    existing_files = {}
    for file_type, file_path in files.items():
        if os.path.exists(file_path):
            existing_files[file_type] = file_path

    # If missing expected files, check fallback names including full variants
    for key, path in fallback.items():
        if os.path.exists(path):
            if 'merged_csv' in key:
                existing_files['merged_csv'] = path
            elif 'csv' in key:
                existing_files['chunked_csv'] = path
            elif 'excel' in key:
                existing_files['chunked_excel'] = path
            elif 'txt' in key:
                existing_files['chunked_txt'] = path

    if not existing_files:
        print(f"No expected files found. Files in {output_folder}:")
        for file in os.listdir(output_folder):
            print(f"  - {file}")
    
    return existing_files

# -------------------------------------------------------------
# Helper: ensure first speaker is Mysteryshopper, second is InsuranceAgent
# -------------------------------------------------------------

def normalize_speaker_names(csv_path):
    """Simple speaker mapping: alternating between Mysteryshopper and InsuranceAgent.
    Row 0: Mysteryshopper, Row 1: InsuranceAgent, Row 2: Mysteryshopper, etc."""
    try:
        df = pd.read_csv(csv_path)
        
        # Simple alternating pattern: even rows = Mysteryshopper, odd rows = InsuranceAgent
        df['Speaker'] = df.index.map(lambda i: 'Mysteryshopper' if i % 2 == 0 else 'InsuranceAgent')
        
        df.to_csv(csv_path, index=False)
        print(f"Applied simple alternating speaker mapping: {len(df)} segments")
        return True
    except Exception as e:
        print(f"Speaker normalization failed for {csv_path}: {e}")
        return False

# -------------------------------------------------------------
# Helper: validate language and flag segments needing manual review
# -------------------------------------------------------------
import re

def validate_language_and_mark(csv_path):
    """Detect whether each Text entry is English or Chinese. If not, mark Label/Notes column."""
    try:
        df = pd.read_csv(csv_path)
        # Ensure a column exists to hold the flag
        label_col = 'Label' if 'Label' in df.columns else ('Notes' if 'Notes' in df.columns else None)
        if label_col is None:
            # Add a new column if missing
            df['Label'] = ''
            label_col = 'Label'
        def is_en_or_zh(text: str) -> bool:
            text = str(text)
            # flag replacement char counts
            if text.count('\uFFFD') / max(1, len(text)) > 0.05:
                return False
            # Accept Chinese
            if re.search(r'[\u4e00-\u9fff]', text):
                return True  # has Chinese characters
            # Remove whitespace for ratio checks
            _clean = re.sub(r'\s+', '', text)
            if not _clean:
                return False
            # English ratio
            letters = re.findall(r'[A-Za-z]', _clean)
            if len(letters) >= len(_clean) * 0.4:
                return True
            # NEW: accept mostly-numeric utterances (e.g., account numbers)
            digits = re.findall(r'\d', _clean)
            if len(digits) >= len(_clean) * 0.6:
                return True
            return False
        df[label_col] = df['Text'].apply(lambda t: '' if is_en_or_zh(t) else 'require human transcription')
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Language validation failed for {csv_path}: {e}")
        return False

# -------------------------------------------------------------
# Helper: create DOCX from chunked CSV
# -------------------------------------------------------------

def generate_docx_from_chunked_csv(csv_path, docx_path):
    """Create a Word document from the chunked CSV (if python-docx is available)."""
    if not DOCX_AVAILABLE:
        return False, "python-docx not installed"
    try:
        import docx
        df = pd.read_csv(csv_path)
        doc = docx.Document()
        doc.add_heading("Conversation Chunks", level=1)

        for idx, row in df.iterrows():
            doc.add_heading(f"Chunk {idx + 1}", level=2)
            doc.add_paragraph(f"Time Range: {row['Start Time']} - {row['End Time']}")
            combined_text = row['Combined Text'].replace('  \n', '\n')
            doc.add_paragraph(combined_text)
            doc.add_page_break()

        doc.save(docx_path)
        return True, "DOCX created"
    except Exception as e:
        return False, str(e)

def create_conversation_chunks(df):
    """
    Creates conversation chunks from a DataFrame.
    Each chunk contains a fixed number of segments (CHUNK_SIZE).
    The combined text includes speaker names.
    """
    CHUNK_SIZE = 5  # Fixed chunk size as requested
    
    # Convert to list of dictionaries for easier processing
    segments = df.to_dict('records')
    
    # Create chunks based on number of segments
    chunks = []
    current_chunk = []
    
    for segment in segments:
        current_chunk.append(segment)
        
        # If we've reached the desired number of segments, start a new chunk
        if len(current_chunk) >= CHUNK_SIZE:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # Create the formatted output data (no Speakers/Label columns)
    formatted_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks, 1):
        # Get chunk start and end times
        start_time = chunk[0]['Start Time']
        end_time = chunk[-1]['End Time']
        
        # Create combined text that already includes speaker names
        combined_text_parts = []
        for segment in chunk:
            speaker = segment['Speaker']
            text = segment['Text']
            combined_text_parts.append(f"{speaker}: {text}")
        
        combined_text = "  \n".join(combined_text_parts)
        
        # Build final row (exclude Speakers/Label per new requirements)
        chunk_record = {
            'Start Time': start_time,
            'End Time': end_time,
            'Combined Text': combined_text,
            'Notes': ''
        }
        
        formatted_chunks.append(chunk_record)
    
    return pd.DataFrame(formatted_chunks)

def run_fullfile_transcription(audio_file, output_dir):
    """
    One-shot call to ElevenLabs for the whole file.
    Produces {base}_full.csv  (+ merged CSV) in output_dir.
    """
    import requests, json

    model = "scribe_v1"
    # Get API key from secrets or environment
    api_key = get_api_key()
    if not api_key:
        return False, "No API key available. Set XI_API_KEY in Streamlit secrets or .env file."  # Early exit
    headers = {"xi-api-key": api_key}
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    with open(audio_file, "rb") as f:
        files = {"file": f}
        data  = {
            "model_id": model,
            "language_code": "en",
            "diarize": True,
            "timestamps_granularity": "word",
            "output_format": "json"
        }
        r = requests.post(url, files=files, data=data, headers=headers)
        ok  = r.status_code == 200
        log = f"Status {r.status_code}\\n{r.text[:500]}..."   # truncate

    if not ok:
        return False, log

    res = r.json()
    segments = []
    cur = None
    for w in res["words"]:
        if w["type"] != "word":
            continue
        sid = w["speaker_id"]
        if cur is None or cur["speaker_id"] != sid:
            if cur:
                segments.append(cur)
            cur = {"speaker_id": sid, "start": w["start"],
                   "end": w["end"], "text": w["text"]}
        else:
            cur["end"]  = w["end"]
            cur["text"] += " " + w["text"]
    if cur:
        segments.append(cur)

    # → DataFrame & CSV
    df = pd.DataFrame([{
        "Start Time": format_timestamp(s["start"]),
        "End Time":   format_timestamp(s["end"]),
        "Speaker":    f"SPEAKER_{s['speaker_id']}",
        "Text":       s["text"],
        "Notes":      ""
    } for s in segments])

    base = Path(audio_file).stem
    csv  = Path(output_dir) / f"{base}_full.csv"
    df.to_csv(csv, index=False)

    # ---- NEW: create merged version similar to elevenlabscribe.py ----
    merged_rows = []
    prev_row = None
    for row in df.to_dict('records'):
        if prev_row is None:
            prev_row = row.copy()
        elif row['Speaker'] == prev_row['Speaker']:
            # Extend previous segment
            prev_row['End Time'] = row['End Time']
            prev_row['Text'] = f"{prev_row['Text']} {row['Text']}"
        else:
            merged_rows.append(prev_row)
            prev_row = row.copy()
    if prev_row is not None:
        merged_rows.append(prev_row)

    merged_df = pd.DataFrame(merged_rows)
    merged_csv = Path(output_dir) / f"{base}_full_merged.csv"
    merged_df.to_csv(merged_csv, index=False)
        
    return True, log

def create_transcript_with_quality_excel(merged_csv_path, quality_results, label_col, output_folder, base_name):
    """Create Excel file with merged transcript and quality analysis logs as separate sheets."""
    import pandas as pd
    import os
    excel_path = os.path.join(output_folder, f"{base_name}_with_quality.xlsx")
    merged_df = pd.read_csv(merged_csv_path)
    
    # Prepare DataFrames for logs
    flagged_df = pd.DataFrame(quality_results['flagged_segments'])
    long_df = pd.DataFrame(quality_results['long_segments'])
    gap_df = pd.DataFrame(quality_results['gap_segments'])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Full transcript
        merged_df.to_excel(writer, sheet_name='Transcript', index=False)
        # Sheet 2: Quality Analysis Logs
        startrow = 0
        # Flagged segments
        if not flagged_df.empty:
            flagged_df[["Start Time", "End Time", "Speaker", "Text", label_col]].to_excel(
                writer, sheet_name='Quality_Logs', index=False, startrow=startrow
            )
            startrow += len(flagged_df) + 2  # +2 for header and one blank row
        # Long segments
        if not long_df.empty:
            long_df[["Start Time", "End Time", "Speaker", "Text", label_col]].to_excel(
                writer, sheet_name='Quality_Logs', index=False, startrow=startrow
            )
            startrow += len(long_df) + 2
        # Gap segments
        if not gap_df.empty:
            gap_df[["Start Time", "End Time", "Speaker", "Text", label_col]].to_excel(
                writer, sheet_name='Quality_Logs', index=False, startrow=startrow
            )
    return excel_path

def process_large_audio_file(audio_file_path, output_folder, st, uploaded_file, progress_bar, status_text):
    """Process a large audio file with progress tracking"""
    try:
        base_name = os.path.splitext(uploaded_file.name)[0]
        
        # Store base_name in session state
        st.session_state['base_name'] = base_name
        
        status_text.text("Running transcription...")
        progress_bar.progress(0.1)

        # Run transcription
        success_tx, tx_logs = run_fullfile_transcription(audio_file_path, output_folder)
        
        # Store transcription logs
        st.session_state.tx_logs = tx_logs
        add_log("Transcription completed", "INFO")
        
        if success_tx:
            progress_bar.progress(0.5)
            status_text.text("Transcription finished – preparing output...")
            
            # Find the merged CSV file
            merged_csv_path = os.path.join(output_folder, f"{base_name}_full_merged.csv")
            
            if os.path.exists(merged_csv_path):
                add_log(f"Found merged CSV: {merged_csv_path}")
                
                # Normalize speaker names
                add_log("Normalizing speaker names...")
                normalize_speaker_names(merged_csv_path)
                
                # Validate language and mark segments
                add_log("Validating language and marking segments...")
                validate_language_and_mark(merged_csv_path)
                
                # Detect gaps
                add_log("Detecting gaps in transcript...")
                gap_count = flag_long_gaps(merged_csv_path, gap_seconds=60)

                # --- Quality Analysis Section ---
                # Read the CSV for analysis
                try:
                    df = pd.read_csv(merged_csv_path)
                    label_col = 'Label' if 'Label' in df.columns else ('Notes' if 'Notes' in df.columns else None)
                    
                    # Store quality analysis results in session state
                    quality_results = {
                        'flagged_segments': [],
                        'long_segments': [],
                        'gap_segments': [],
                        'total_issues': 0,
                        'total_segments': len(df)
                    }
                    
                    if label_col:
                        # Check for gibberish/unsupported language
                        flagged = df[df[label_col].str.contains('require human transcription', na=False)]
                        if not flagged.empty:
                            quality_results['flagged_segments'] = flagged.to_dict('records')
                        
                        # Check for long segments (>120s)
                        long_segments = df[df[label_col].str.contains('>120s single segment', na=False)]
                        if not long_segments.empty:
                            quality_results['long_segments'] = long_segments.to_dict('records')
                        
                        # Check for gaps
                        gaps = df[df[label_col].str.contains('>60s silence', na=False)]
                        if gap_count > 0:
                            quality_results['gap_segments'] = gaps.to_dict('records')
                        
                        # Summary
                        total_issues = len(flagged) + len(long_segments) + gap_count
                        quality_results['total_issues'] = total_issues
                    
                    # Store quality results in session state
                    st.session_state['quality_results'] = quality_results
                    st.session_state['quality_df'] = df.to_dict('records')
                    st.session_state['label_col'] = label_col
                
                except Exception as e:
                    add_log(f"Quality analysis failed: {str(e)}", "ERROR")
                
                # Create output files dictionary
                output_files = {'merged_csv': merged_csv_path}
                
                excel_with_quality = create_transcript_with_quality_excel(merged_csv_path, quality_results, label_col, output_folder, base_name)
                output_files['excel_with_quality'] = excel_with_quality

                st.success("🎉 Processing completed successfully!")

                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zf:
                    # Only add the new Excel file
                    zf.write(excel_with_quality, os.path.basename(excel_with_quality))
                zip_buffer.seek(0)

                # Store results in session state
                st.session_state['zip_data'] = zip_buffer.getvalue()
                st.session_state['output_files'] = output_files
                # Store preview data
                if 'excel_with_quality' in output_files:
                    with open(output_files['excel_with_quality'], 'rb') as f:
                        st.session_state['excel_with_quality_bytes'] = f.read()

                # Download button
                st.download_button(
                    label="⬇️ Download Transcript + Quality Logs (Excel in ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"{base_name}_transcript_quality.zip",
                    mime="application/zip",
                    type="primary",
                    key="download_button"
                )

                # Show preview
                if 'merged_csv' in output_files:
                    st.subheader("Merged Transcript Preview")
                    df_preview = pd.read_csv(output_files['merged_csv']).head(10)
                    st.dataframe(df_preview, use_container_width=True)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                add_log("Processing completed successfully!")
                
            else:
                error_msg = "Merged CSV file not found after transcription"
                st.error(f"❌ {error_msg}")
                add_log(error_msg, "ERROR")
        else:
            error_msg = "Transcription failed"
            st.error(f"❌ {error_msg}")
            add_log(error_msg, "ERROR")
            
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        add_log(error_msg, "ERROR")
        add_log(traceback.format_exc(), "ERROR")
        raise

# -------------------------------------------------------------
# Helper: flag long gaps with no transcription
# -------------------------------------------------------------

def flag_long_gaps(csv_path, gap_seconds: int = 60):
    """Add a warning to Label/Notes if there is a silent gap longer than gap_seconds
    between consecutive segments.
    """
    try:
        import pandas as pd
        import re
        
        df = pd.read_csv(csv_path)
        label_col = 'Label' if 'Label' in df.columns else ('Notes' if 'Notes' in df.columns else None)
        if label_col is None:
            df['Label'] = ''
            label_col = 'Label'

        # Parse timestamps 00:01:20,519 -> timedelta
        def to_timedelta(t):
            try:
                t = t.strip()
                # allow comma or . between seconds and ms and optional spaces after comma
                m = re.match(r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})[,.]\s*(?P<ms>\d{1,3})", t)
                if not m:
                    return float('nan')
                total = int(m['h'])*3600 + int(m['m'])*60 + int(m['s']) + int(m['ms'])/1000
                return total
            except Exception:
                return None

        # Helper to append messages safely
        def add_flag(series, mask, flag_text):
            def _append(val):
                base = '' if (pd.isna(val) or str(val).strip()=='') else f"{str(val).strip()}; "
                return base + flag_text
            # Use .loc with proper assignment to avoid warning
            series_copy = series.copy()
            series_copy.loc[mask] = series_copy.loc[mask].apply(_append)
            return series_copy

        starts = df['Start Time'].apply(lambda x: to_timedelta(str(x)))
        prev_ends = df['End Time'].shift().apply(lambda x: to_timedelta(str(x)))
        gaps = starts - prev_ends
        df['__gap__'] = gaps
        # First row will be NaN; ignore
        gap_mask = df['__gap__'] > gap_seconds
        if gap_mask.any():
            df[label_col] = add_flag(df[label_col], gap_mask, '>60s silence, missing audio segment')
        # After gap detection modifications within flag_long_gaps
        # Extra: mark empty or whitespace-only text
        empty_mask = df['Text'].apply(lambda x: str(x).strip() == '')
        if empty_mask.any():
            df[label_col] = add_flag(df[label_col], empty_mask, 'empty text, missing transcription')
        # Flag rows whose own duration > 120 s (possible untranscribed part)
        durations = df.apply(lambda r: to_timedelta(str(r['End Time'])) - to_timedelta(str(r['Start Time'])), axis=1)
        longgap_mask = durations > 120
        if longgap_mask.any():
            df[label_col] = add_flag(df[label_col], longgap_mask, '>120s single segment, possible missing transcription')
        df.drop(columns='__gap__', inplace=True)
        df.to_csv(csv_path, index=False)
        return gap_mask.sum()
    except Exception as e:
        print(f"Gap validation failed for {csv_path}: {e}")
        return 0

# -------------------------------------------------------------
# Helper: detect and flag 90+ second gaps in chunked CSV
# -------------------------------------------------------------

def detect_large_gaps_in_transcript(csv_path, min_gap_seconds=90):
    """Detect gaps longer than min_gap_seconds in the transcript that may indicate missing transcription"""
    try:
        df = pd.read_csv(csv_path)
        
        def to_seconds(timestamp_str):
            """Convert HH:MM:SS,mmm to seconds"""
            try:
                time_part, ms_part = timestamp_str.split(',')
                h, m, s = map(int, time_part.split(':'))
                ms = int(ms_part)
                return h * 3600 + m * 60 + s + ms / 1000
            except:
                return 0
        
        large_gaps = []
        
        for i in range(len(df) - 1):
            current_end = to_seconds(df.iloc[i]['End Time'])
            next_start = to_seconds(df.iloc[i + 1]['Start Time'])
            gap_duration = next_start - current_end
            
            if gap_duration >= min_gap_seconds:
                large_gaps.append({
                    'Gap Number': len(large_gaps) + 1,
                    'After Segment': i + 1,
                    'Gap Start': df.iloc[i]['End Time'],
                    'Gap End': df.iloc[i + 1]['Start Time'],
                    'Duration': format_duration(gap_duration),
                    'duration_seconds': gap_duration,
                    'Previous Text': df.iloc[i]['Text'][:50] + "..." if len(df.iloc[i]['Text']) > 50 else df.iloc[i]['Text'],
                    'Next Text': df.iloc[i + 1]['Text'][:50] + "..." if len(df.iloc[i + 1]['Text']) > 50 else df.iloc[i + 1]['Text']
                })
        
        return {
            'gaps_found': len(large_gaps) > 0,
            'large_gaps': large_gaps,
            'total_gaps': len(large_gaps)
        }
    except Exception as e:
        return {
            'gaps_found': False,
            'large_gaps': [],
            'total_gaps': 0,
            'error': str(e)
        }

def format_duration(seconds):
    """Format seconds into readable duration (e.g., '2m 30s' or '1h 15m 30s')"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if secs > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{hours}h"

def detect_90s_gaps_in_chunked_csv(csv_path):
    """
    Detect chunks longer than 90 seconds in the chunked CSV file, which may indicate
    missing transcription within the chunk. Print timestamps and flag them in Notes.
    Returns count of long chunks found and log messages.
    """
    log_messages = []
    try:
        df = pd.read_csv(csv_path)
        
        # Parse timestamps to seconds
        def to_seconds(timestamp_str):
            try:
                timestamp_str = str(timestamp_str).strip().strip('"')
                import re
                # Match HH:MM:SS,mmm or HH:MM:SS.mmm format
                match = re.match(r'(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})', timestamp_str)
                if match:
                    hours, minutes, seconds, milliseconds = match.groups()
                    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
                    return total_seconds
                return None
            except:
                return None
        
        # Convert timestamps to seconds
        df['start_seconds'] = df['Start Time'].apply(to_seconds)
        df['end_seconds'] = df['End Time'].apply(to_seconds)
        
        # Calculate duration of each chunk
        long_chunks_found = []
        long_chunk_count = 0
        
        for i in range(len(df)):
            start_sec = df.iloc[i]['start_seconds']
            end_sec = df.iloc[i]['end_seconds']
            
            if start_sec is not None and end_sec is not None:
                chunk_duration = end_sec - start_sec
                
                if chunk_duration >= 90:  # 90+ second chunk
                    long_chunk_count += 1
                    start_time = df.iloc[i]['Start Time']
                    end_time = df.iloc[i]['End Time']
                    
                    # Format chunk duration
                    duration_minutes = int(chunk_duration // 60)
                    duration_remainder = int(chunk_duration % 60)
                    duration_str = f"{duration_minutes}m {duration_remainder}s" if duration_minutes > 0 else f"{duration_remainder}s"
                    
                    chunk_info = {
                        'chunk_number': long_chunk_count,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration_str,
                        'duration_seconds': chunk_duration
                    }
                    long_chunks_found.append(chunk_info)
                    
                    # Print and log the chunk information
                    log_msg = f"🚨 Long Chunk #{long_chunk_count}: {duration_str} chunk from {start_time} to {end_time}"
                    print(log_msg)
                    log_messages.append(log_msg)
                    
                    # Flag in Notes column
                    current_notes = df.iloc[i]['Notes']
                    if pd.isna(current_notes) or str(current_notes).strip() == '' or current_notes == 'nan':
                        flag_text = f"Long chunk ({duration_str}) - may contain missing transcription"
                    else:
                        flag_text = f"{str(current_notes).strip()}; Long chunk ({duration_str}) - may contain missing transcription"
                    
                    df.iloc[i, df.columns.get_loc('Notes')] = flag_text
        
        # Save the updated CSV with flags
        df.drop(columns=['start_seconds', 'end_seconds'], inplace=True)
        df.to_csv(csv_path, index=False)
        
        # Print and log summary
        if long_chunk_count > 0:
            summary_msg = f"📊 Summary: Found {long_chunk_count} chunk(s) longer than 90 seconds"
            print(summary_msg)
            log_messages.append(summary_msg)
            for chunk in long_chunks_found:
                detail_msg = f"   • Chunk {chunk['chunk_number']}: {chunk['duration']} from {chunk['start_time']} to {chunk['end_time']}"
                print(detail_msg)
                log_messages.append(detail_msg)
        else:
            success_msg = "✅ No chunks longer than 90 seconds found"
            print(success_msg)
            log_messages.append(success_msg)
        
        return long_chunk_count, long_chunks_found, log_messages
        
    except Exception as e:
        error_msg = f"❌ Error detecting long chunks in chunked CSV {csv_path}: {e}"
        print(error_msg)
        log_messages.append(error_msg)
        return 0, [], log_messages

# -------------------------------------------------------------

def display_quality_analysis():
    """Display the persistent quality analysis results"""
    if hasattr(st.session_state, 'quality_results') and st.session_state.quality_results:
        st.subheader("🔍 Transcript Quality Analysis")
        
        quality_results = st.session_state.quality_results
        label_col = st.session_state.get('label_col', 'Notes')
        
        # Display flagged segments
        if quality_results['flagged_segments']:
            st.warning(f"⚠️ {len(quality_results['flagged_segments'])} segment(s) were flagged as requiring human transcription due to gibberish or unsupported language.")
            with st.expander("🔍 View flagged segments"):
                df_flagged = pd.DataFrame(quality_results['flagged_segments'])
                st.dataframe(df_flagged[['Start Time', 'End Time', 'Speaker', 'Text', label_col]], use_container_width=True)
        
        # Display long segments
        if quality_results['long_segments']:
            st.error(f"🚨 Found {len(quality_results['long_segments'])} segment(s) longer than 120 seconds - possible missing transcription!")
            with st.expander("🔍 View long segments"):
                df_long = pd.DataFrame(quality_results['long_segments'])
                st.dataframe(df_long[['Start Time', 'End Time', 'Speaker', 'Text', label_col]], use_container_width=True)
        
        # Display gap segments
        if quality_results['gap_segments']:
            st.warning(f"⏳ Found {len(quality_results['gap_segments'])} segment(s) with gaps > 60s")
            with st.expander("🔍 View segments with gaps"):
                df_gaps = pd.DataFrame(quality_results['gap_segments'])
                st.dataframe(df_gaps[['Start Time', 'End Time', 'Speaker', 'Text', label_col]], use_container_width=True)
        
        # Display summary
        if quality_results['total_issues'] == 0:
            st.success("✅ No quality issues detected!")
        else:
            st.info(f"📊 Quality Summary: {quality_results['total_issues']} total issues detected across {quality_results['total_segments']} segments")

def main():
    st.set_page_config(page_title="Intage Audio Transcription UI", layout="wide")
    
    # Initialize logging
    init_logging()
    
    st.title("Intage Audio Transcription UI")
    
    # File uploader section
    st.subheader("Choose an audio file")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Limit 500MB per file • WAV, MP3, FLAC, OGG, M4A"
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            temp_audio_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Processing method selection
            st.subheader("Processing method:")
            transcription_mode = st.radio(
                "Select transcription mode",
                ["Transcribe whole file"],
                label_visibility="collapsed"
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if st.button("🎯 Start Transcription", type="primary"):
                try:
                    process_large_audio_file(temp_audio_path, output_dir, st, uploaded_file, progress_bar, status_text)
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    add_log(f"Processing failed: {str(e)}", "ERROR")
                    add_log(f"Traceback: {traceback.format_exc()}", "ERROR")
            
            # Display quality analysis if it exists (persists after download)
            if hasattr(st.session_state, 'quality_results') and st.session_state.quality_results:
                display_quality_analysis()
            
            # Clear session button
            if st.button("🗑️ Clear session & temp files"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()

if __name__ == "__main__":
    main() 