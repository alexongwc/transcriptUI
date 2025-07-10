#!/bin/bash

# ElevenLabs Audio Transcription Streamlit App Launcher

echo "🎙️ Starting ElevenLabs Audio Transcription App..."
echo "========================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if required Python packages are available
python3 -c "import streamlit, soundfile, pandas, requests, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not found. Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All dependencies found"
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will be available at:"
echo "  Local URL: http://localhost:8502"
echo "  Network URL: http://10.254.50.50:8502"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
streamlit run streamlit_app.py --server.port 8502 --server.address 0.0.0.0 