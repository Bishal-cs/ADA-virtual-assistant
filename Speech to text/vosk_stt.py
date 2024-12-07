import os
import sys
import vosk
import json
import queue
import numpy as np
import sounddevice as sd

MODEL_PATH = "Speech to text/STT models/vosk-model-small-en-us-0.15"

if not os.path.exists(MODEL_PATH):
    print(f"Model path '{MODEL_PATH}' does not exist !!!. Please download again.")
    sys.exit(1)

model = vosk.Model(MODEL_PATH)

q = queue.Queue()

sample_rate = 16000
CHANNELS = 1

def callback(indata, frames, time, status):
    """Callback function for capture audio data from mic"""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def process_audio():
    """Function to process incoming data"""
    rec = vosk.KaldiRecognizer(model, sample_rate)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and 'ADA' in result['text'].lower():
                print("Welcome in ADA")
                break

def start_audio_stream():
    """Function to start audio stream"""
    with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=sample_rate, dtype='int16'):
        print("Listening...")
        process_audio()

def main():
    try:
        start_audio_stream()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()