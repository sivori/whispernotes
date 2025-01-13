# WhisperNotes

Real-time meeting transcription tool that uses OpenAI's Whisper for speech recognition and optional speaker diarization.

## Features

- Real-time audio transcription using Whisper's tiny model
- Automatic hourly file organization
- Timestamp markers for easy reference
- Optional speaker diarization (requires Hugging Face token)
- CPU-optimized for broad compatibility
- Graceful start/stop with Ctrl+C

## Installation

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install install numpy torch pyaudio openai-whisper pyannote.audio python-dotenv

```

4. System Dependencies:
   - **Windows**: No additional steps needed
   - **macOS**:
     ```bash
     brew install portaudio
     ```
   - **Linux**:
     ```bash
     sudo apt-get install python3-pyaudio portaudio19-dev
     ```

5. (Optional) Speaker Diarization Setup:
   - Visit https://huggingface.co/pyannote/speaker-diarization
   - Accept the user agreement
   - Create an access token at https://hf.co/settings/tokens
   - Create a `.env` file in the project root:
     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```

## Usage

1. Start transcription:

```bash
python meeting_transcriber.py
```

2. Press Ctrl+C to stop recording

## Output

- Transcripts are saved in the `meeting_notes` directory
- Files are organized by hour: `meeting_notes_YYYYMMDD_HH00.txt`
- Each file includes:
  - Session start/end times
  - Minute markers
  - Speaker labels (if diarization is enabled)
  - Transcribed text

## Notes

- The tool uses Whisper's "tiny" model for faster processing
- All processing is done locally on CPU
- Speaker diarization is optional and requires additional setup
- Audio is processed in ~3-second chunks for real-time feedback

## Troubleshooting

If you encounter issues:

1. **Audio Input**: Ensure your microphone is properly connected and selected as default input device
2. **Speaker Diarization**: Verify your Hugging Face token and user agreement acceptance
3. **Performance**: If transcription is slow, ensure no other intensive processes are running

## License

[Your chosen license]
