import whisper
import pyaudio
import numpy as np
import threading
import queue
import datetime
from pathlib import Path
import signal
import sys

class MeetingTranscriber:
    def __init__(self):
        # Use tiny model instead of base, and force CPU mode with lower precision
        try:
            self.model = whisper.load_model("tiny", device="cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.max_audio_buffer = self.RATE * 3

        # Initialize state
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.threads = []
        
        # File handling
        self.output_dir = Path("meeting_notes")
        self.output_dir.mkdir(exist_ok=True)
        self.current_hour = datetime.datetime.now().strftime("%Y%m%d_%H")
        self.current_file = None
        self.file_lock = threading.Lock()

    def _open_new_file(self):
        with self.file_lock:
            if self.current_file:
                try:
                    self.current_file.close()
                except:
                    pass
            filename = self.output_dir / f"meeting_notes_{self.current_hour}00.txt"
            self.current_file = open(filename, "a")
            self.current_file.write(f"\nSession started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _save_transcription(self, text):
        try:
            with self.file_lock:
                current_time = datetime.datetime.now()
                new_hour = current_time.strftime("%Y%m%d_%H")
                
                # Check if we need to start a new hourly file
                if new_hour != self.current_hour:
                    self.current_hour = new_hour
                    self._open_new_file()
                
                # Open file if not already open
                if not self.current_file or self.current_file.closed:
                    self._open_new_file()
                
                # Write the transcription with timestamp
                timestamp = current_time.strftime("%H:%M:%S")
                self.current_file.write(f"[{timestamp}] {text}\n")
                self.current_file.flush()
                print(f"Transcribed: {text}")
        except Exception as e:
            print(f"Error saving transcription: {e}")

    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def _transcribe_audio(self):
        audio_buffer = []
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                while len(audio_buffer) < self.max_audio_buffer and (self.is_recording or not self.audio_queue.empty()):
                    if not self.audio_queue.empty():
                        data = self.audio_queue.get(timeout=1)
                        audio_buffer.extend(np.frombuffer(data, dtype=np.float32))
                
                if len(audio_buffer) > 0:
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    result = self.model.transcribe(
                        audio_array,
                        fp16=False,
                        language="en",
                        without_timestamps=True
                    )
                    
                    if result["text"].strip():
                        self._save_transcription(result["text"])
                    
                    audio_buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during transcription: {e}")
                audio_buffer = []
                continue

    def start_recording(self):
        self._open_new_file()
        self.is_recording = True
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Start threads
        self.threads = [
            threading.Thread(target=self._record_audio),
            threading.Thread(target=self._transcribe_audio)
        ]
        for thread in self.threads:
            thread.start()

    def stop_recording(self):
        print("\nStopping recording...")
        self.is_recording = False
        
        # Clean up stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Clean up file
        with self.file_lock:
            if self.current_file and not self.current_file.closed:
                self.current_file.write("\nSession ended at "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.current_file.close()
        
        # Clean up PyAudio
        self.audio.terminate()

def signal_handler(signum, frame):
    print("\nSignal received, stopping gracefully...")
    if 'transcriber' in globals():
        transcriber.stop_recording()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    transcriber = MeetingTranscriber()
    print("Starting meeting transcription (press Ctrl+C to stop)...")
    try:
        transcriber.start_recording()
        signal.pause()  # Wait for signal
    except KeyboardInterrupt:
        transcriber.stop_recording() 