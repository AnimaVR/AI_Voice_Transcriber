import pyaudio
import wave
import keyboard
import base64
import requests
import json
import os
from datetime import datetime
import threading

# Configuration
SERVER_URL = "http://192.168.1.1:6969/transcribe"
TRANSCRIPTION_FOLDER = "transcriptions"
AUDIO_FILE = "recorded_audio.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper's default
CHUNK = 1024

# Ensure the transcription folder exists
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)

def generate_unique_filename():
    """Generate a unique filename based on the current date and time."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(TRANSCRIPTION_FOLDER, f"transcription_{timestamp}.txt")

def record_audio(stop_signal):
    """Record audio using pyaudio until the stop_signal is set."""
    p = pyaudio.PyAudio()

    # Create audio stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Recording started...")
    while not stop_signal.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return AUDIO_FILE

def audio_to_base64(file_path):
    """Convert audio file to base64."""
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64

def send_to_api(audio_base64):
    """Send base64 audio to the Whisper API and return the transcription."""
    headers = {"Content-Type": "application/json"}
    data = {"audio_base64": audio_base64}

    try:
        response = requests.post(SERVER_URL, json=data, headers=headers)
        response.raise_for_status()
        json_response = response.json()

        if "transcription" in json_response:
            return json_response["transcription"]
        else:
            return "Error: Transcription not found in response."
    except requests.RequestException as e:
        return f"Error: {e}"

def save_transcription(transcription):
    """Save transcription to a uniquely named text file in the transcription folder."""
    unique_filename = generate_unique_filename()
    with open(unique_filename, "w") as f:
        f.write(transcription)
    print(f"Transcription saved to {unique_filename}")

if __name__ == "__main__":
    print("Press and hold Right-Ctrl to record. Press Space to toggle start/stop recording. Press Esc to quit.")

    stop_signal = threading.Event()
    space_mode_active = False

    while True:
        try:
            # Space mode: Press once to start and again to stop
            if keyboard.is_pressed("space"):
                if not space_mode_active:  # Start recording
                    space_mode_active = True
                    stop_signal.clear()
                    audio_thread = threading.Thread(target=record_audio, args=(stop_signal,))
                    audio_thread.start()
                    keyboard.wait("space")  # Wait for the user to press space again
                    stop_signal.set()
                    audio_thread.join()  # Wait for the recording thread to finish

                    # Process the audio
                    audio_path = AUDIO_FILE
                    audio_base64 = audio_to_base64(audio_path)
                    transcription = send_to_api(audio_base64)
                    save_transcription(transcription)
                    space_mode_active = False

            # Right-Ctrl mode: Hold to record
            elif keyboard.is_pressed("right ctrl"):
                stop_signal.clear()
                audio_path = record_audio(stop_signal)

                # Process the audio
                audio_base64 = audio_to_base64(audio_path)
                transcription = send_to_api(audio_base64)
                save_transcription(transcription)

            # Exit program
            elif keyboard.is_pressed("esc"):
                print("\nProgram exited.")
                break

        except KeyboardInterrupt:
            print("\nProgram exited.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
