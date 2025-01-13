import whisper
import pyaudio
import wave
import numpy as np
import threading
import queue
import datetime
from pathlib import Path

class MeetingTranscriber:
    def __init__(self):
        # Use tiny model instead of base, and force CPU mode with lower precision
        try:
            self.model = whisper.load_model("tiny", device="cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Reduce buffer size to prevent memory issues
        self.CHUNK = 1024  # Reduced from 1024 * 4
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        
        # Add memory-efficient processing
        self.max_audio_buffer = self.RATE * 3  # Process 3 seconds at a time instead of 5

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Create output directory for notes
        self.output_dir = Path("meeting_notes")
        self.output_dir.mkdir(exist_ok=True)

    def start_recording(self):
        self.is_recording = True
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Start recording thread
        threading.Thread(target=self._record_audio).start()
        # Start transcription thread
        threading.Thread(target=self._transcribe_audio).start()

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def _transcribe_audio(self):
        audio_buffer = []
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Process smaller chunks of audio
                while len(audio_buffer) < self.max_audio_buffer and (self.is_recording or not self.audio_queue.empty()):
                    if not self.audio_queue.empty():
                        data = self.audio_queue.get()
                        audio_buffer.extend(np.frombuffer(data, dtype=np.float32))

                if len(audio_buffer) > 0:
                    # Convert to float32 numpy array
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    
                    # Use more memory-efficient transcription options
                    result = self.model.transcribe(
                        audio_array,
                        fp16=False,  # Disable FP16 to prevent CPU issues
                        language="en",  # Specify language if known
                        without_timestamps=True  # Disable timestamp generation to save memory
                    )
                    
                    if result["text"].strip():
                        self._save_transcription(result["text"])
                    
                    # Clear buffer
                    audio_buffer = []
                    
            except Exception as e:
                print(f"Error during transcription: {e}")
                continue

    def _save_transcription(self, text):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"meeting_notes_{timestamp}.txt"
        
        with open(filename, "a") as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {text}\n")
            print(f"Transcribed: {text}")

# Example usage
if __name__ == "__main__":
    transcriber = MeetingTranscriber()
    
    print("Starting meeting transcription (press Ctrl+C to stop)...")
    try:
        transcriber.start_recording()
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        transcriber.stop_recording() 