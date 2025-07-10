# Audio Transcription UI

A Streamlit-based web application for transcribing audio files using ElevenLabs API with speaker identification and conversation analysis.

## Features

- 🎵 **Audio Transcription**: Transcribe audio files using ElevenLabs API
- 👥 **Speaker Identification**: Automatically identify and label speakers
- 📊 **Conversation Chunking**: Break conversations into manageable segments
- 🚨 **Gap Detection**: Detect and flag missing transcription segments longer than 90 seconds
- 📝 **Multiple Export Formats**: Export results as CSV, Excel, TXT, or Word documents
- 🔍 **Quality Control**: Detect and flag gibberish or low-quality transcription segments
- 📈 **Processing Logs**: Detailed logs of the transcription process

## Demo Access

This application is ready to use! The API key is pre-configured for demo purposes.

**Live Demo:** [Access the application here](https://intageaudio.streamlit.app)

## Local Development Setup (Optional)

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd audioUI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (for local development):**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your ElevenLabs API key:
   ```
   XI_API_KEY=your_elevenlabs_api_key_here
   ```

4. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. **Upload Audio File**: Drag and drop or select an audio file (WAV, MP3, FLAC, OGG, M4A)
2. **Choose Processing Options**: Select speaker identification and chunking preferences
3. **Start Transcription**: Click "Start Transcription" to begin processing
4. **Review Results**: View the transcription with speaker labels and conversation chunks
5. **Download Results**: Export the results in your preferred format

## Configuration

The application uses environment variables for configuration:

- `XI_API_KEY`: Your ElevenLabs API key (required)
- `MAX_DURATION`: Maximum duration for conversation chunks in seconds (default: 300)
- `SEGMENTS_PER_CHUNK`: Number of segments per conversation chunk (default: 5)

## Features in Detail

### Speaker Identification
- Automatically identifies speakers in conversations
- Maps speakers to meaningful labels (Mysteryshopper, InsuranceAgent)
- Supports multiple speaker identification formats

### Gap Detection
- Detects conversation segments longer than 90 seconds
- Flags potential missing transcription areas
- Provides detailed timestamp information

### Quality Control
- Detects gibberish or low-quality transcription segments
- Flags segments that may need manual review
- Provides confidence scoring for transcription quality

### Export Options
- **CSV**: Structured data with timestamps and speaker labels
- **Excel**: Formatted spreadsheet with multiple sheets
- **TXT**: Plain text format for easy reading
- **Word**: Professional document format

## File Structure

```
audioUI/
├── streamlit_app.py      # Main Streamlit application
├── elevenlabscribe.py    # ElevenLabs API integration
├── chunk.py              # Conversation chunking logic
├── config_safe.py        # Safe configuration with environment variables
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- The `.env` file is excluded from git by default
- API keys are masked in logs and UI displays

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for internal use. Please ensure compliance with ElevenLabs API terms of service.

## Support

For questions or issues, please contact the development team or create an issue in the repository. 