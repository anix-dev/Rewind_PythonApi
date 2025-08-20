from fastapi import APIRouter, UploadFile, File, HTTPException
from pydub import AudioSegment
import wave
import os
import json
from vosk import Model, KaldiRecognizer

router = APIRouter()

# Correct path to Vosk model (relative to project root)
model_path = os.path.join("app", "models", "speech", "en-us")

if not os.path.exists(model_path):
    raise RuntimeError(f"Vosk model not found at {model_path}. Make sure it's downloaded and extracted.")
model = Model(model_path)


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file, converts it to 16kHz mono WAV,
    and transcribes speech using Vosk.
    """
    try:
        # Save uploaded file temporarily
        temp_input = "temp_input.m4a"
        with open(temp_input, "wb") as f:
            f.write(await file.read())

        # Convert to wav (mono, 16kHz)
        temp_wav = "temp.wav"
        audio = AudioSegment.from_file(temp_input)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav, format="wav")

        # Run Vosk recognition
        wf = wave.open(temp_wav, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        result_text = ""

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result_text += rec.Result()
        result_text += rec.FinalResult()

        wf.close()
        os.remove(temp_input)
        os.remove(temp_wav)

        return json.loads(result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
