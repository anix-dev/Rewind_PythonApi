from fastapi import APIRouter, UploadFile, File, HTTPException
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator
import io, wave, json

router = APIRouter(tags=["Transcription"])

# Use only Hindi small model for multilingual recognition
model_path = "app/models/speech/hi"
model = Model(model_path)

def safe_json_parse(text: str) -> str:
    """
    Safely parse Vosk JSON result, return empty string on failure.
    """
    try:
        data = json.loads(text)
        return data.get("text", "")
    except json.JSONDecodeError:
        return ""

@router.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != "audio":
        raise HTTPException(status_code=400, detail="Invalid audio file")

    try:
        # Convert audio to mono 16kHz WAV
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        wf = wave.open(wav_io, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())

        result_text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result_text.append(safe_json_parse(rec.Result()))

        result_text.append(safe_json_parse(rec.FinalResult()))

        # Merge text
        text = " ".join([r for r in result_text if r.strip() != ""])

        if not text:
            raise HTTPException(status_code=500, detail="Transcription failed: No text recognized")

        # Translate to English
        translation = GoogleTranslator(source='auto', target='en').translate(text)

        return {"transcription_en": translation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
