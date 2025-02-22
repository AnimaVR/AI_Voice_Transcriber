from flask import Flask, request, jsonify
from transformers import pipeline
import soundfile as sf
import resampy
import numpy as np
from base64 import b64decode
from io import BytesIO

app = Flask(__name__)

# Initialize the ASR pipeline on GPU (if available)
generator = pipeline(
    task="automatic-speech-recognition", 
    model="whisper-large-v3", 
    device=0
)

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        audio_base64 = request.json.get('audio_base64')
        if not audio_base64:
            return jsonify({"status": "error", "message": "No audio provided."}), 400
        
        # Decode the base64 audio and read it using soundfile
        audio_bytes = b64decode(audio_base64)
        audio_input, original_sampling_rate = sf.read(BytesIO(audio_bytes))
        
        # If stereo, convert to mono
        if len(audio_input.shape) == 2:
            audio_input = np.mean(audio_input, axis=1)
        
        # Resample to 16,000 Hz if needed
        target_sampling_rate = 16000
        if original_sampling_rate != target_sampling_rate:
            audio_input = resampy.resample(audio_input, original_sampling_rate, target_sampling_rate)
        
        # Determine audio duration
        audio_duration = len(audio_input) / target_sampling_rate
        # If longer than 29 seconds, enable timestamps
        return_timestamps = audio_duration > 29
        
        # Transcribe with or without timestamps based on audio duration
        transcription = generator(audio_input, return_timestamps=return_timestamps)
        
        if transcription and isinstance(transcription, dict):
            transcription_text = transcription.get('text', '').strip()
            response_data = {"status": "success", "transcription": transcription_text}
            if return_timestamps:
                response_data['timestamps'] = transcription.get('segments', [])
            return jsonify(response_data)
        else:
            return jsonify({
                "status": "error", 
                "message": "Transcription failed or unexpected format."
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010)
