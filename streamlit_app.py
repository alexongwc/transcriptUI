import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

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
        # Get the absolute path to the audioUI directory
        audioui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(audioui_dir, "elevenlabscribe.py")
        
        # Create a temporary modified version of elevenlabscribe.py
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Replace the hardcoded paths with absolute paths
        # Build path strings
        abs_audio = os.path.abspath(audio_file_path)
        abs_out   = os.path.abspath(output_folder)

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
            st.success("✅ Transcription completed successfully!")
            return True, log_text
        else:
            st.error("❌ Transcription failed")
            return False, log_text
            
    except Exception as e:
        st.error(f"❌ Error running transcription: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, f"Exception: {str(e)}"

def run_chunking(csv_file_path, output_folder):
    """Run the chunk.py script to create conversation chunks"""
    try:
        # Get the absolute path to the audioUI directory
        audioui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(audioui_dir, "chunk.py")
        
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
        
        # Run the script from the audioUI directory
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, cwd=audioui_dir)
        
        # Clean up temp script
        os.remove(temp_script)
        
        log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        if result.returncode == 0:
            st.success("✅ Chunking completed successfully!")
            return True, log_text
        else:
            st.error("❌ Chunking failed")
            return False, log_text
            
    except Exception as e:
        st.error(f"❌ Error running chunking: {str(e)}")
        import traceback
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

def process_large_audio_file(audio_file_path, output_folder, st, uploaded_file, progress_bar, status_text):
    """
    Processes a large audio file by chunking it and then running transcription and chunking.
    This function is called when the uploaded file exceeds the 200MB limit.
    """
    st.warning(f"Processing large audio file ({len(uploaded_file.getvalue()) / (1024*1024):.1f} MB) in chunks.")
    progress_bar.progress(0.05)
    status_text.text("Saving uploaded file...")
    base_name = os.path.splitext(uploaded_file.name)[0]
    # The file is already saved by the caller (audio_file_path)
    # Create output directory in audioUI
    audioui_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(audioui_dir, "temp_output")
    os.makedirs(output_folder, exist_ok=True)
    status_text.text("Running transcription on chunks/whole file...")
    progress_bar.progress(0.2)
    # Step 1: Run full file transcription (to get initial segments)
    success_tx, tx_logs = run_fullfile_transcription(audio_file_path, output_folder)
    if success_tx:
        progress_bar.progress(0.5)
        status_text.text("Transcription finished – post-processing...")
        # Step 2: Find the merged CSV file (full-file path always produces _full_merged.csv)
        merged_csv_path = os.path.join(output_folder, f"{base_name}_full_merged.csv")
        
        if os.path.exists(merged_csv_path):
            # Normalize speaker names so first speaker is Mysteryshopper
            normalize_speaker_names(merged_csv_path)
            validate_language_and_mark(merged_csv_path)
            gap_count = flag_long_gaps(merged_csv_path, gap_seconds=60)

            # --- NEW: summarize flagged (gibberish) segments ---
            try:
                _df_flag = pd.read_csv(merged_csv_path)
                _label_col = 'Label' if 'Label' in _df_flag.columns else ('Notes' if 'Notes' in _df_flag.columns else None)
                if _label_col:
                    _flagged = _df_flag[_df_flag[_label_col].str.contains('require human transcription', na=False)]
                    if not _flagged.empty:
                        st.warning(f"⚠️ {_flagged.shape[0]} segment(s) were flagged as requiring human transcription due to gibberish or unsupported language.")
                        with st.expander("🔍 View flagged segments"):
                            st.dataframe(_flagged.head(20), use_container_width=True)
                    # long-gap flag summary
                    _gaps = _df_flag[_df_flag[_label_col].str.contains('>60s silence', na=False)]
                    if gap_count:
                        st.warning(f"⏳ Detected {gap_count} segment(s) flagged as '>60s silence, missing audio segment'.")
                        with st.expander("🔍 View long-gap segments"):
                            st.dataframe(_gaps[['Start Time','End Time','Speaker','Text',_label_col]].head(20), use_container_width=True)
            except Exception as _e:
                st.info(f"Language-validation summary unavailable: {_e}")
            progress_bar.progress(0.6)
            status_text.text("Creating conversation chunks...")
            # Step 3: Run chunking
            success_ck, ck_logs = run_chunking(merged_csv_path, output_folder)
            if success_ck:
                st.success("✅ Processing complete!")
                progress_bar.progress(1.0)
                # Find all output files (use base_name, not base_name_merged)
                output_files = find_output_files(output_folder, base_name)

                # Check for 90+ second long chunks in chunked CSV
                if 'chunked_csv' in output_files:
                    st.info("🔍 Checking for 90+ second long chunks in transcript...")
                    chunk_count, long_chunks_found, gap_log_messages = detect_90s_gaps_in_chunked_csv(output_files['chunked_csv'])
                    
                    if chunk_count > 0:
                        st.warning(f"⏳ Found {chunk_count} chunk(s) longer than 90 seconds (may contain missing transcription)")
                        with st.expander("🔍 View long chunks"):
                            chunk_df = pd.DataFrame(long_chunks_found)
                            st.dataframe(chunk_df[['chunk_number', 'start_time', 'end_time', 'duration']], use_container_width=True)
                    else:
                        st.success("✅ No chunks longer than 90 seconds found")

                # Generate DOCX if chunked CSV exists
                if 'chunked_csv' in output_files:
                    docx_path = os.path.join(output_folder, f"{base_name}_chunked.docx")
                    success_docx, msg_docx = generate_docx_from_chunked_csv(output_files['chunked_csv'], docx_path)
                    if success_docx:
                        output_files['chunked_docx'] = docx_path
                    else:
                        st.warning(f"DOCX not generated: {msg_docx}")
                
                if output_files:
                    st.success("🎉 Processing completed successfully!")

                    # Create a single ZIP with all output files
                    import zipfile, io
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        for file_type, file_path in output_files.items():
                            zf.write(file_path, os.path.basename(file_path))
                    zip_buffer.seek(0)

                    st.download_button(
                        label="⬇️ Download All Results (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"{base_name}_results.zip",
                        mime="application/zip",
                        type="primary",
                        key="large_file_download_button"
                    )

                    # Cache results in session_state so they persist on rerun
                    st.session_state['zip_data'] = zip_buffer.getvalue()
                    st.session_state['output_files'] = output_files
                    st.session_state['base_name'] = base_name
                    # store preview bytes
                    if 'merged_csv' in output_files:
                        with open(output_files['merged_csv'], 'rb') as _f:
                            st.session_state['merged_csv_bytes'] = _f.read()

                    # Show a preview of merged CSV (first 10 rows)
                    if 'merged_csv' in output_files:
                        st.subheader("Merged Transcript Preview")
                        df_preview = pd.read_csv(output_files['merged_csv']).head(10)
                        st.dataframe(df_preview, use_container_width=True)

                    # Show logs in expander
                    with st.expander("🔍 View Processing Logs"):
                        st.subheader("Transcription Logs")
                        st.text(tx_logs)
                        st.subheader("Chunking Logs")
                        st.text(ck_logs)
                        if 'gap_log_messages' in locals() and gap_log_messages:
                            st.subheader("Gap Detection Logs")
                            st.text("\n".join(gap_log_messages))
                else:
                    st.error("❌ No output files found")
                    # Debug: Show what files are in the output folder
                    st.write("Files in output folder:")
                    for file in os.listdir(output_folder):
                        st.write(f"- {file}")
            else:
                st.error("❌ Chunking failed")
                with st.expander("🔍 Transcription Logs"):
                    st.text(tx_logs)
        else:
            st.error("❌ Merged CSV file not found after transcription")
            # Debug: Show what files are in the output folder
            st.write("Files in output folder:")
            for file in os.listdir(output_folder):
                st.write(f"- {file}")
    else:
        st.error("❌ Transcription failed")
        with st.expander("🔍 Transcription Logs"):
            st.text(tx_logs)
    # Keep temp files so preview/download still work. They can be cleared via a button in main().

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

def main():
    st.set_page_config(
        page_title="Intage Audio Transcription",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Intage Audio Transcription")
    st.markdown("Upload an audio file to get transcription with speaker identification and conversation chunks")
    
    # Check API key availability
    api_key = get_api_key()
    if not api_key:
        st.error("🔑 **Configuration Error**")
        st.markdown("Please contact support - API configuration is missing.")
        st.stop()
    else:
        st.success("✅ Ready for transcription")

    # Show existing results if present
    if 'zip_data' in st.session_state and 'output_files' in st.session_state:
        st.success("🎉 Processing completed successfully! (cached)")
        st.download_button(
            label="⬇️ Download All Results (ZIP)",
            data=st.session_state['zip_data'],
            file_name=f"{st.session_state.get('base_name','results')}_results.zip",
            mime="application/zip",
            type="primary",
            key="cached_download_button"
        )
        if 'merged_csv_bytes' in st.session_state:
            import io
            st.subheader("Merged Transcript Preview")
            df_preview = pd.read_csv(io.BytesIO(st.session_state['merged_csv_bytes'])).head(10)
            st.dataframe(df_preview, use_container_width=True)
        st.divider()
        if st.button("🗑️ Clear session & temp files"):
            st.session_state.clear()
            try:
                shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_output"), ignore_errors=True)
            except Exception:
                pass
            st.rerun()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A (Max 500MB)"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024*1024)
        if file_size_mb > 200:
            st.warning(f"File size ({file_size_mb:.1f} MB) exceeds the 200MB limit. Large files may take longer.")
        
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Create output directory in audioUI
        audioui_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(audioui_dir, "temp_output")
        os.makedirs(output_folder, exist_ok=True)
        
        base_name = os.path.splitext(uploaded_file.name)[0]
        
        # Save uploaded file to audioUI directory
        audio_file_path = os.path.join(output_folder, uploaded_file.name)
        with open(audio_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Method selector
            method = st.selectbox(
                "Processing method:",
                (
                    "Transcribe whole file (send whole file to Elevenlabs)",
                )
            )

            if st.button("🚀 Start Transcription", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Running transcription...")
                progress_bar.progress(0.1)

                if method.startswith("Method 1"):
                    success_tx, tx_logs = run_elevenlabs_transcription(
                        audio_file_path, output_folder)
                else:
                    success_tx, tx_logs = run_fullfile_transcription(
                        audio_file_path, output_folder)

                if success_tx:
                    progress_bar.progress(0.5)
                    status_text.text("Transcription finished – preparing chunks...")
                    
                    # Step 2: Find the merged CSV file
                    if method.startswith("Method 1"):
                        merged_csv_path = os.path.join(output_folder, f"{base_name}_merged.csv")
                    else:
                        merged_csv_path = os.path.join(output_folder, f"{base_name}_full_merged.csv")
 
                    if os.path.exists(merged_csv_path):
                        # Normalize speaker names so first speaker is Mysteryshopper
                        normalize_speaker_names(merged_csv_path)
                        # Validate language and mark segments needing manual review
                        validate_language_and_mark(merged_csv_path)
                        gap_count = flag_long_gaps(merged_csv_path, gap_seconds=60)

                        # --- Gap Detection & Quality Analysis ---
                        st.subheader("🔍 Transcript Quality Analysis")
                        try:
                            _df_flag = pd.read_csv(merged_csv_path)
                            _label_col = 'Label' if 'Label' in _df_flag.columns else ('Notes' if 'Notes' in _df_flag.columns else None)
                            if _label_col:
                                # Language/gibberish flags
                                _flagged = _df_flag[_df_flag[_label_col].str.contains('require human transcription', na=False)]
                                if not _flagged.empty:
                                    st.warning(f"⚠️ {_flagged.shape[0]} segment(s) were flagged as requiring human transcription due to gibberish or unsupported language.")
                                    with st.expander("🔍 View flagged segments"):
                                        st.dataframe(_flagged.head(20), use_container_width=True)
                                
                                # 60s+ silence gaps
                                _gaps = _df_flag[_df_flag[_label_col].str.contains('>60s silence', na=False)]
                                if gap_count:
                                    st.warning(f"⏳ Detected {gap_count} segment(s) flagged as '>60s silence, missing audio segment'.")
                                    with st.expander("🔍 View long-gap segments"):
                                        st.dataframe(_gaps[['Start Time','End Time','Speaker','Text',_label_col]].head(20), use_container_width=True)
                                
                                # 120s+ single segments (possible missing transcription)
                                _long_segments = _df_flag[_df_flag[_label_col].str.contains('>120s single segment', na=False)]
                                if not _long_segments.empty:
                                    st.error(f"🚨 Found {_long_segments.shape[0]} segment(s) longer than 120 seconds - possible missing transcription!")
                                    with st.expander("🔍 View long segments"):
                                        st.dataframe(_long_segments[['Start Time','End Time','Speaker','Text',_label_col]].head(20), use_container_width=True)
                                
                                # Summary
                                total_issues = len(_flagged) + gap_count + len(_long_segments)
                                if total_issues == 0:
                                    st.success("✅ No quality issues detected - transcript appears complete and accurate!")
                                else:
                                    st.info(f"📊 Quality Summary: {total_issues} total issues detected across {len(_df_flag)} segments")
                                    
                        except Exception as _e:
                            st.info(f"Quality analysis unavailable: {_e}")
                        status_text.text("Creating conversation chunks...")
                        progress_bar.progress(0.8)
                        
                        # Step 3: Run chunking (optional)
                        output_files = {}
                        if ENABLE_CHUNKING:
                            success_ck, ck_logs = run_chunking(merged_csv_path, output_folder)
                            if success_ck:
                                progress_bar.progress(1.0)
                                status_text.text("✅ Processing complete!")
                                output_files = find_output_files(output_folder, base_name)
                        else:
                            # No chunking – just use merged CSV as the only output
                            success_ck, ck_logs = True, ""  # pretend success so no error banner
                            progress_bar.progress(1.0)
                            status_text.text("✅ Processing complete! (chunking disabled)")
                            output_files['merged_csv'] = merged_csv_path

                        # Display results and provide download
                        if output_files:
                            st.success("🎉 Processing completed successfully!")

                            # Create a single ZIP with all output files
                            import zipfile, io
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                                for file_type, file_path in output_files.items():
                                    zf.write(file_path, os.path.basename(file_path))
                            zip_buffer.seek(0)

                            st.download_button(
                                label="⬇️ Download All Results (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"{base_name}_results.zip",
                                mime="application/zip",
                                type="primary",
                                key="main_download_button"
                            )

                            # Cache results in session_state so they persist on rerun
                            st.session_state['zip_data'] = zip_buffer.getvalue()
                            st.session_state['output_files'] = output_files
                            st.session_state['base_name'] = base_name
                            # store preview bytes
                            if 'merged_csv' in output_files:
                                with open(output_files['merged_csv'], 'rb') as _f:
                                    st.session_state['merged_csv_bytes'] = _f.read()

                            # Optional checks only if chunking is enabled and produced CSV
                            if ENABLE_CHUNKING and 'chunked_csv' in output_files:
                                # Check for long gaps
                                st.info("🔍 Checking for 90+ second long chunks in transcript...")
                                chunk_count, long_chunks_found, gap_log_messages = detect_90s_gaps_in_chunked_csv(output_files['chunked_csv'])
                                if chunk_count > 0:
                                    st.warning(f"⏳ Found {chunk_count} chunk(s) longer than 90 seconds (may contain missing transcription)")
                                    with st.expander("🔍 View long chunks"):
                                        chunk_df = pd.DataFrame(long_chunks_found)
                                        st.dataframe(chunk_df[['chunk_number', 'start_time', 'end_time', 'duration']], use_container_width=True)

                                # Generate DOCX
                                docx_path = os.path.join(output_folder, f"{base_name}_chunked.docx")
                                success_docx, msg_docx = generate_docx_from_chunked_csv(output_files['chunked_csv'], docx_path)
                                if success_docx:
                                    output_files['chunked_docx'] = docx_path

                                # Show preview of chunked CSV (5 segments per row) if available
                                st.subheader("🎯 Chunked Transcript Preview (5 segments per row)")
                                st.success("✅ This is the chunked format you requested - 5 conversation segments combined into 1 row!")
                                df_chunked_preview = pd.read_csv(output_files['chunked_csv']).head(5)
                                st.dataframe(df_chunked_preview, use_container_width=True)
                            


                            # Show logs in expander
                            with st.expander("🔍 View Processing Logs"):
                                st.subheader("Transcription Logs")
                                st.text(tx_logs)
                                if ENABLE_CHUNKING:
                                    st.subheader("Chunking Logs")
                                    st.text(ck_logs)
                                if 'gap_log_messages' in locals() and gap_log_messages:
                                    st.subheader("Gap Detection Logs")
                                    st.text("\n".join(gap_log_messages))
                        else:
                            st.error("❌ No output files found")
                            # Debug: Show what files are in the output folder
                            st.write("Files in output folder:")
                            for file in os.listdir(output_folder):
                                st.write(f"- {file}")
                    else:
                        st.error("❌ Merged CSV file not found after transcription")
                        # Debug: Show what files are in the output folder
                        st.write("Files in output folder:")
                        for file in os.listdir(output_folder):
                            st.write(f"- {file}")
                else:
                    st.error("❌ Transcription failed")
                    with st.expander("🔍 Transcription Logs"):
                        st.text(tx_logs)
            
        finally:
            # keep temp files for session caching so user can preview/download later
            pass

if __name__ == "__main__":
    main() 