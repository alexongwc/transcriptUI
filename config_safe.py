"""
Configuration file for Audio Transcription App
Uses environment variables for sensitive data
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ElevenLabs API Key - NEVER hardcode this!
API_KEY = os.getenv("XI_API_KEY")
if not API_KEY:
    raise ValueError(
        "XI_API_KEY environment variable is required. "
        "Please set it in your .env file or environment variables."
    )

# Audio processing settings
MAX_DURATION = int(os.getenv("MAX_DURATION", "300"))  # 5 minutes default
SEGMENTS_PER_CHUNK = int(os.getenv("SEGMENTS_PER_CHUNK", "5"))

# Speaker mapping configuration
SPEAKER_MAPPING = {
    "SPEAKER_speaker_0": "Mysteryshopper",
    "SPEAKER_speaker_1": "InsuranceAgent",
    "SPEAKER_0": "Mysteryshopper", 
    "SPEAKER_1": "InsuranceAgent",
    "speaker_0": "Mysteryshopper",
    "speaker_1": "InsuranceAgent"
}

# File processing settings
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
MAX_FILE_SIZE_MB = 500

# Output settings
OUTPUT_FORMATS = ['csv', 'xlsx', 'txt', 'docx']

print("✅ Configuration loaded successfully!")
if API_KEY:
    print(f"   API Key: {API_KEY[:10]}...{API_KEY[-4:]} (length: {len(API_KEY)})")
else:
    print("   ⚠️  API Key: Not set!") 