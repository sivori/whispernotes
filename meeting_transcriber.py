import whisper
import pyaudio
import numpy as np
import threading
import queue
import datetime
from pathlib import Path
import signal
import sys
import torch
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

class MeetingTranscriber:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
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
        self.last_timestamp = datetime.datetime.now().replace(second=0, microsecond=0)

        # Initialize speaker diarization pipeline with better error handling
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                print("\nWarning: HUGGINGFACE_TOKEN not found in .env file")
                print("Speaker diarization will be disabled\n")
                print("To enable speaker detection:")
                print("1. Visit https://huggingface.co/pyannote/speaker-diarization")
                print("2. Accept the user agreement")
                print("3. Create an access token at https://hf.co/settings/tokens")
                print("4. Add your token to .env file as HUGGINGFACE_TOKEN=your_token_here\n")
                self.diarization = None
            else:
                try:
                    self.diarization = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization",
                        use_auth_token=hf_token
                    )
                    self.diarization = self.diarization.to(torch.device("cpu"))
                except Exception as e:
                    print("\nError: Could not load speaker diarization model")
                    print("Please make sure you have:")
                    print("1. Accepted the user agreement at https://huggingface.co/pyannote/speaker-diarization")
                    print("2. Used a valid access token")
                    print(f"Original error: {str(e)}\n")
                    self.diarization = None
        except Exception as e:
            print(f"Error initializing diarization: {e}")
            self.diarization = None

        # Add current speaker tracking
        self.current_speaker = None

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

    def _save_transcription(self, text, speaker=None):
        try:
            with self.file_lock:
                current_time = datetime.datetime.now()
                new_hour = current_time.strftime("%Y%m%d_%H")
                
                # Check if we need to start a new hourly file
                if new_hour != self.current_hour:
                    self.current_hour = new_hour
                    self._open_new_file()
                    self.current_speaker = None  # Reset speaker on new file
                
                # Open file if not already open
                if not self.current_file or self.current_file.closed:
                    self._open_new_file()
                
                # Check if we need to add a minute timestamp
                current_minute = current_time.replace(second=0, microsecond=0)
                if current_minute > self.last_timestamp:
                    minute_marker = current_minute.strftime("%H:%M:00")
                    self.current_file.write(f"\n----- {minute_marker} -----\n\n")
                    self.last_timestamp = current_minute
                
                # Only write speaker tag if it changed
                if speaker and speaker != self.current_speaker:
                    self.current_file.write(f"\n[{speaker}]\n")
                    self.current_speaker = speaker
                    print(f"Speaker changed to: {speaker}")
                
                # Write the transcription
                self.current_file.write(f"{text}\n")
                print(f"Transcribed: {text}")
                    
                self.current_file.flush()
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
                    
                    # Only attempt diarization if it's available
                    speaker = None
                    if self.diarization:
                        try:
                            # Convert numpy array to torch tensor in correct format (channel, time)
                            waveform = torch.from_numpy(audio_array).unsqueeze(0)  # Add channel dimension
                            diarization = self.diarization({
                                "waveform": waveform,
                                "sample_rate": self.RATE
                            })
                            for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                                speaker = f"SPEAKER_{speaker_id}"
                                break
                        except Exception as e:
                            print(f"Diarization error (continuing without speaker detection): {e}")
                    
                    # Transcribe as before
                    result = self.model.transcribe(
                        audio_array,
                        fp16=False,
                        language="en",
                        without_timestamps=True
                    )
                    
                    if result["text"].strip():
                        self._save_transcription(result["text"], speaker if speaker else None)
                    
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