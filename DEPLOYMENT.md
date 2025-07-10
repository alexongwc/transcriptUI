# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repository with your code
- ElevenLabs API key
- Streamlit Cloud account

## Step-by-Step Deployment

### 1. Prepare Your Repository
✅ Your repository is already configured with:
- `.gitignore` excluding sensitive files
- `requirements.txt` with all dependencies
- Environment variable configuration
- Streamlit configuration files

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app:**
   - Repository: `alexongwc/transcriptUI`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: Choose a custom URL (e.g., `your-app-name`)

### 3. Add Secrets (API Key)

✅ **API Key Already Configured**
The API key is pre-configured in the repository secrets file for demo purposes.

If you need to update it:
1. **After deployment, go to your app settings**
2. **Click on "Secrets"**
3. **Update the secrets:**
   ```toml
   XI_API_KEY = "your_actual_elevenlabs_api_key_here"
   ```
4. **Save the secrets**

### 4. Test Your Deployment

1. **Visit your app URL**
2. **You should see "✅ API Key loaded" message**
3. **Upload a test audio file**
4. **Verify transcription works**

## Configuration Files

### `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 500
headless = true
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

### Secrets Configuration
The app will automatically detect if it's running on Streamlit Cloud and use `st.secrets` for the API key. For local development, it falls back to environment variables.

## Features Available in Deployment

✅ **Audio Upload** - Up to 500MB files
✅ **Speaker Identification** - Automatic speaker labeling
✅ **Conversation Chunking** - Organized conversation segments
✅ **Gap Detection** - Identifies missing transcription segments
✅ **Multiple Export Formats** - CSV, Excel, TXT, Word
✅ **Quality Control** - Gibberish detection and flagging
✅ **Real-time Processing** - Progress indicators and logs

## Troubleshooting

### Common Issues:

1. **API Key Not Found**
   - Check that secrets are properly configured in Streamlit Cloud
   - Ensure the secret name is exactly `XI_API_KEY`

2. **File Upload Issues**
   - Maximum file size is 500MB
   - Supported formats: WAV, MP3, FLAC, OGG, M4A

3. **Processing Errors**
   - Check the processing logs in the expandable section
   - Verify your ElevenLabs API key has sufficient credits

### Performance Notes:
- Large files (>200MB) may take several minutes to process
- The app uses ElevenLabs' speech-to-text API which has usage limits
- Processing time depends on audio length and complexity

## Security

✅ **API Key Security**
- API keys are stored securely in Streamlit Cloud secrets
- Keys are never exposed in logs or UI (only last 4 characters shown)
- No sensitive data is committed to version control

✅ **File Security**
- Uploaded files are processed in temporary directories
- Files are automatically cleaned up after processing
- No permanent storage of user files

## Support

For issues with the deployment:
1. Check the Streamlit Cloud logs
2. Verify all secrets are properly configured
3. Ensure your ElevenLabs API key is valid and has credits
4. Contact support if needed

## App URL
After deployment, your app will be available at:
`https://your-app-name.streamlit.app` 