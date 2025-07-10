import pandas as pd
import os

# Configuration
INPUT_CSV = "/home/alexong/intage/1476_JunKaiOng_GEFA_m2_merged.csv"
OUTPUT_FOLDER = "/home/alexong/intage"
SEGMENTS_PER_CHUNK = 5  # Number of segments per chunk (like in the image)

def create_folder_if_not_exists(folder_path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_formatted_chunks(csv_file_path):
    """Create chunks in the exact format shown in the user's CSV files"""
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Input file not found: {csv_file_path}")
        return
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    print(f"Total segments: {len(df)}")
    
    # Convert to list of dictionaries for easier processing
    segments = df.to_dict('records')
    
    # Create chunks based on number of segments
    chunks = []
    current_chunk = []
    
    for segment in segments:
        current_chunk.append(segment)
        
        # If we've reached the desired number of segments, start a new chunk
        if len(current_chunk) >= SEGMENTS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} conversation chunks")
    
    # Create the formatted output data
    formatted_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks, 1):
        # Get chunk start and end times
        start_time = chunk[0]['Start Time']
        end_time = chunk[-1]['End Time']
        
        # Create combined text which already includes speaker names
        combined_text_parts = []
        note_parts = []
        for segment in chunk:
            speaker = segment['Speaker']
            text = segment['Text']
            # Format like: "Speaker: text"
            combined_text_parts.append(f"{speaker}: {text}")
            # capture any Label or Notes from original segment
            for note_field in ('Label','Notes'):
                if note_field in segment and str(segment[note_field]).strip():
                    note_parts.append(str(segment[note_field]).strip())
        
        combined_text = "  \n".join(combined_text_parts)

        notes_combined = "; ".join(sorted(set(note_parts))) if note_parts else ''

        # Create the chunk record (exclude Label and Speakers columns per new requirements)
        chunk_record = {
            'Start Time': start_time,
            'End Time': end_time,
            'Combined Text': combined_text,
            'Notes': notes_combined
        }
        
        formatted_chunks.append(chunk_record)
    
    # Create DataFrame
    chunks_df = pd.DataFrame(formatted_chunks)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    # Save as Excel file
    excel_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.xlsx")
    chunks_df.to_excel(excel_file, index=False, engine='openpyxl')
    
    # Save as CSV file (similar to Excel but in CSV format)
    csv_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.csv")
    chunks_df.to_csv(csv_file, index=False)
    
    # Save as TXT file with readable format
    txt_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunked.txt")
    
    with open(txt_file, 'w') as f:
        f.write("Conversation Chunks - Formatted Output\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source: {csv_file_path}\n")
        f.write(f"Total segments: {len(df)}\n")
        f.write(f"Segments per chunk: {SEGMENTS_PER_CHUNK}\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in chunks_df.iterrows():
            f.write(f"CHUNK {idx + 1:02d}\n")
            f.write("-" * 15 + "\n")
            f.write(f"Time Range: {row['Start Time']} - {row['End Time']}\n")
            f.write(f"Notes: {row['Notes']}\n\n")
            
            f.write("Combined Text:\n")
            # Replace the line breaks for better TXT readability
            combined_text = row['Combined Text'].replace('  \n', '\n')
            f.write(combined_text)
            f.write("\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"\n✅ Formatted chunks created!")
    print(f"Excel file: {excel_file}")
    print(f"CSV file: {csv_file}")
    print(f"TXT file: {txt_file}")
    
    return excel_file, csv_file, txt_file

def create_combined_chunks_file(csv_file_path):
    """Create one combined file with all chunks together"""
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Input file not found: {csv_file_path}")
        return
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    print(f"Total segments: {len(df)}")
    
    # Convert to list of dictionaries for easier processing
    segments = df.to_dict('records')
    
    # Create chunks based on number of segments
    chunks = []
    current_chunk = []
    
    for segment in segments:
        current_chunk.append(segment)
        
        # If we've reached the desired number of segments, start a new chunk
        if len(current_chunk) >= SEGMENTS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} conversation chunks")
    
    # Create the combined file
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    combined_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_all_chunks.txt")
    
    with open(combined_file, 'w') as f:
        f.write("All Conversation Chunks Combined\n")
        f.write("=================================\n\n")
        f.write(f"Source: {csv_file_path}\n")
        f.write(f"Total segments: {len(df)}\n")
        f.write(f"Segments per chunk: {SEGMENTS_PER_CHUNK}\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            # Write chunk header
            f.write(f"CHUNK {i:02d}\n")
            f.write("-" * 10 + "\n")
            f.write(f"Time Range: {chunk[0]['Start Time']} - {chunk[-1]['End Time']}\n")
            f.write(f"Speakers: {', '.join(set(segment['Speaker'] for segment in chunk))}\n")
            f.write(f"Segments: {len(chunk)}\n\n")
            
            # Write combined text
            f.write("Combined Text:\n")
            combined_text = []
            for segment in chunk:
                speaker = segment['Speaker']
                text = segment['Text']
                combined_text.append(f"{speaker}: {text}")
            
            f.write(" ".join(combined_text))
            f.write("\n\n")
            
            # Write individual segments
            f.write("Individual Segments:\n")
            for segment in chunk:
                speaker = segment['Speaker']
                text = segment['Text']
                start_time = segment['Start Time']
                end_time = segment['End Time']
                f.write(f"[{start_time} - {end_time}] {speaker}: {text}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n✅ Combined chunks file created!")
    print(f"File: {combined_file}")
    return combined_file

def chunk_conversations(csv_file_path):
    """Chunk conversations into groups of segments"""
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Input file not found: {csv_file_path}")
        return
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    print(f"Total segments: {len(df)}")
    
    # Convert to list of dictionaries for easier processing
    segments = df.to_dict('records')
    
    # Create chunks based on number of segments
    chunks = []
    current_chunk = []
    
    for i, segment in enumerate(segments):
        current_chunk.append(segment)
        
        # If we've reached the desired number of segments, start a new chunk
        if len(current_chunk) >= SEGMENTS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} conversation chunks")
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_dir = os.path.join(OUTPUT_FOLDER, f"{base_name}_chunks")
    create_folder_if_not_exists(output_dir)
    
    # Save each chunk as a separate CSV file
    for i, chunk in enumerate(chunks, 1):
        chunk_df = pd.DataFrame(chunk)
        chunk_file = os.path.join(output_dir, f"chunk_{i:02d}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        
        # Calculate chunk statistics
        duration_start = chunk[0]['Start Time']
        duration_end = chunk[-1]['End Time']
        
        # Count unique speakers in chunk
        speakers = set(segment['Speaker'] for segment in chunk)
        speaker_list = ', '.join(speakers)
        
        print(f"Chunk {i:2d}: {len(chunk):2d} segments, {duration_start} - {duration_end}, Speakers: {speaker_list}")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "chunk_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Conversation Chunking Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Input file: {csv_file_path}\n")
        f.write(f"Total segments: {len(df)}\n")
        f.write(f"Segments per chunk: {SEGMENTS_PER_CHUNK}\n")
        f.write(f"Total chunks created: {len(chunks)}\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            duration_start = chunk[0]['Start Time']
            duration_end = chunk[-1]['End Time']
            speakers = set(segment['Speaker'] for segment in chunk)
            speaker_list = ', '.join(speakers)
            f.write(f"Chunk {i:2d}: {len(chunk):2d} segments, {duration_start} - {duration_end}, Speakers: {speaker_list}\n")
    
    print(f"\n✅ Chunking complete!")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    
    return output_dir

def create_combined_text_chunks(csv_file_path):
    """Create text-only chunks for easy reading with combined text format"""
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Input file not found: {csv_file_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    segments = df.to_dict('records')
    
    # Create chunks based on number of segments
    chunks = []
    current_chunk = []
    
    for segment in segments:
        current_chunk.append(segment)
        
        # If we've reached the desired number of segments, start a new chunk
        if len(current_chunk) >= SEGMENTS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # Create text-only chunks with combined format
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_dir = os.path.join(OUTPUT_FOLDER, f"{base_name}_text_chunks")
    create_folder_if_not_exists(output_dir)
    
    for i, chunk in enumerate(chunks, 1):
        text_file = os.path.join(output_dir, f"chunk_{i:02d}.txt")
        
        with open(text_file, 'w') as f:
            f.write(f"Conversation Chunk {i}\n")
            f.write(f"===================\n")
            f.write(f"Time Range: {chunk[0]['Start Time']} - {chunk[-1]['End Time']}\n")
            f.write(f"Speakers: {', '.join(set(segment['Speaker'] for segment in chunk))}\n\n")
            
            # Create combined text like in the image
            combined_text = []
            for segment in chunk:
                speaker = segment['Speaker']
                text = segment['Text']
                combined_text.append(f"{speaker}: {text}")
            
            f.write("Combined Text:\n")
            f.write(" ".join(combined_text))
            f.write(f"\n\n")
            
            # Also show individual segments
            f.write("Individual Segments:\n")
            f.write("-" * 20 + "\n")
            for segment in chunk:
                speaker = segment['Speaker']
                text = segment['Text']
                start_time = segment['Start Time']
                end_time = segment['End Time']
                f.write(f"[{start_time} - {end_time}] {speaker}: {text}\n")
            
            f.write(f"\n--- End of Chunk {i} ---\n")
    
    print(f"Text chunks created in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    print("Starting conversation chunking...")
    print(f"Input file: {INPUT_CSV}")
    print(f"Segments per chunk: {SEGMENTS_PER_CHUNK}")
    print("-" * 50)
    
    # Create the formatted chunks (Excel, CSV, TXT) like the user's example
    excel_file, csv_file, txt_file = create_formatted_chunks(INPUT_CSV)
    
    print(f"\n✅ Formatted files created!")
    print(f"Excel: {excel_file}")
    print(f"CSV: {csv_file}")
    print(f"TXT: {txt_file}")
    
    # Optionally create other formats
    # combined_file = create_combined_chunks_file(INPUT_CSV)
    # chunk_dir = chunk_conversations(INPUT_CSV)
    # text_dir = create_combined_text_chunks(INPUT_CSV) 