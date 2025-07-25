# Transcription Analysis Tool Usage Guide

## Overview
The transcription analysis tool (`transcription_analysis.py`) provides comprehensive analysis of transcription results from the iFlytek API, including performance metrics, quality assessment, and advanced post-processing capabilities.

## How to Use

### Basic Usage
```bash
python transcription_analysis.py <json_file>
```

### Available Options
- `--summary` or `-s`: Print performance summary to console
- `--output <file>` or `-o <file>`: Save detailed analysis to JSON file
- `--reference <file>` or `-r <file>`: Provide reference text for Word Error Rate calculation
- `--help` or `-h`: Show help message

### Examples

#### 1. Quick Performance Summary
```bash
python transcription_analysis.py transcription_results/transcription_20250714_103125.json --summary
```

**Output:**
```
================================================================================
🎙️  TRANSCRIPTION PERFORMANCE ANALYSIS
================================================================================
📊 Basic Metrics:
  • Total Duration: 920.8 seconds
  • Processing Time: 0:15:37.797119
  • Real-time Factor: 1.02x
  • Total Segments: 150
  • Segments per Minute: 9.8

👥 Speaker Analysis:
  • mysteryshopper: 81 segments (54.0%)
  • insuranceagent: 69 segments (46.0%)
  • Speaker Balance: 0.85

📝 Content Analysis:
  • Total Words: 939
  • Average Words per Segment: 6.3
  • Average Confidence Score: 0.00

🌐 Language Analysis:
  • English: 104 segments
  • Numbers: 15 segments
  • Punctuation: 94 segments
  • Chinese: 62 segments
  • Mixed Language Segments: 23 (15.3%)

💬 Conversation Flow:
  • Speaker Switches: 116
  • Average Turn Length: 39.8 segments
  • Quick Switches: 20
  • Interruption Rate: 13.3%
  • Conversation Pace: 9.8 segments/minute
```

#### 2. Detailed Analysis Report
```bash
python transcription_analysis.py transcription_results/transcription_20250714_103125.json -o detailed_report.json
```

This creates a comprehensive JSON report with:
- Performance metrics
- Language analysis
- Conversation flow analysis
- Extracted key information (numbers, dates, etc.)
- Quality assessment

#### 3. Word Error Rate Calculation
```bash
python transcription_analysis.py transcription_results/transcription_20250714_103125.json -r reference_text.txt
```

## Understanding the Outputs

### 1. Performance Metrics

#### Basic Metrics
- **Total Duration**: Length of audio in seconds
- **Processing Time**: Time taken to process the audio
- **Real-time Factor**: Processing time / Audio duration (1.0 = real-time)
- **Total Segments**: Number of transcription segments
- **Segments per Minute**: Transcription density

#### Speaker Analysis
- **Speaker Distribution**: How much each speaker talked
- **Speaker Balance**: Ratio of least to most talkative speaker (1.0 = perfect balance)
- **Dominant Speaker**: Who talked the most

#### Content Analysis
- **Word/Character Counts**: Total and average per segment
- **Confidence Scores**: Average confidence of transcription
- **Quality Metrics**: Empty segments, short/long segments

### 2. Language Analysis

#### Language Distribution
- **English**: English language segments
- **Chinese**: Chinese language segments  
- **Numbers**: Numeric content
- **Punctuation**: Punctuation marks

#### Mixed Language Detection
- **Mixed Segments**: Segments containing multiple languages
- **Mixing Percentage**: % of segments with language mixing
- **Examples**: Sample mixed language segments

### 3. Conversation Flow

#### Turn-taking Analysis
- **Speaker Switches**: Number of times speakers changed
- **Average Turn Length**: Average segments per speaker turn
- **Quick Switches**: Rapid speaker changes (< 2 seconds)
- **Interruption Rate**: % of quick switches
- **Conversation Pace**: Segments per minute

### 4. Extracted Information

#### Key Data Extraction
- **Numbers**: Phone numbers, amounts, dates
- **Emails**: Email addresses found
- **Dates**: Date references
- **Context**: Surrounding text for each extraction

### 5. Quality Assessment

#### Quality Metrics
- **Segments with Confidence**: Segments with confidence scores
- **Empty Segments**: Segments with no text
- **Short Segments**: Segments with < 3 words
- **Long Segments**: Segments with > 20 words

## Input File Format

The tool expects a JSON file with this structure:
```json
{
  "session_info": {
    "timestamp": "2025-07-14T10:31:25.254832",
    "audio_file": "converted_audio.wav",
    "processing_time": "0:15:37.797119"
  },
  "segments": [
    {
      "timestamp": "00:00:06.919",
      "speaker": "mysteryshopper",
      "text": "This is dream car today is september 2024 is around 10 am. I'm meeting",
      "raw_data": [...]
    }
  ]
}
```

## Output File Format

The detailed analysis report includes:
```json
{
  "analysis_timestamp": "2025-07-14T10:56:57.165352",
  "source_file": "transcription_results/transcription_20250714_103125.json",
  "performance_metrics": {
    "basic_metrics": {...},
    "speaker_analysis": {...},
    "content_analysis": {...},
    "quality_metrics": {...}
  },
  "language_analysis": {...},
  "conversation_flow": {...},
  "extracted_information": {...}
}
```

## Use Cases

### 1. Call Center Quality Assessment
- Monitor speaker balance
- Detect interruptions
- Measure conversation flow
- Extract key information (phone numbers, amounts)

### 2. Language Learning Analysis
- Track language mixing patterns
- Measure vocabulary usage
- Analyze conversation structure

### 3. Performance Optimization
- Monitor real-time factor
- Identify processing bottlenecks
- Track confidence scores

### 4. Content Analysis
- Extract structured data
- Identify key topics
- Measure engagement metrics

## Tips for Best Results

1. **Use High-Quality Audio**: Better audio = better transcription = better analysis
2. **Check Speaker Labels**: Ensure speaker diarization is accurate
3. **Review Confidence Scores**: Low confidence may indicate transcription errors
4. **Consider Context**: Some metrics may vary by conversation type
5. **Compare Multiple Files**: Use for benchmarking and trend analysis

## Troubleshooting

### Common Issues
- **File Not Found**: Ensure JSON file path is correct
- **Invalid JSON**: Check that transcription file is properly formatted
- **No Segments**: Verify transcription contains actual content
- **Memory Issues**: Large files may require more memory

### Performance Notes
- Analysis time scales with file size
- Large files (>100MB) may take several minutes
- Consider using summary mode for quick checks 