from fastapi import File, UploadFile, HTTPException, APIRouter, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
import soundfile as sf
import librosa
from io import BytesIO
import tempfile
import os
from src.utils.logger import logger
from src.services.process import transcribe_audio
from gtts import gTTS
from pydub import AudioSegment
import uuid
from src.models.pydantic_model import RequestText

router = APIRouter(prefix="/api", tags=["TRANSCRIBE APIS"])






@router.post("/text-to-wav/")
async def text_to_wav(request:RequestText):
    """
    Convert text to speech and return WAV audio file.
    """
    uid = str(uuid.uuid4())
    mp3_path = f"/tmp/{uid}.mp3"
    wav_path = f"/tmp/{uid}.wav"

    try:
        # Generate MP3 file from text
        tts = gTTS(request.text)
        tts.save(mp3_path)

        # Convert MP3 to WAV
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")

        # Read wav content into memory for streaming
        wav_bytes = open(wav_path, "rb").read()

    finally:
        # Clean up temp files
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # Stream WAV file as response without saving permanently on disk
    return StreamingResponse(BytesIO(wav_bytes), media_type="audio/wav", headers={
        "Content-Disposition": f"attachment; filename=output_{uid}.wav"
    })

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accept a WAV file upload and return transcription text.
    """
    # Validate file extension
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    contents = await file.read()

    # Validate WAV file and duration
    try:
        audio_bytes_io = BytesIO(contents)
        audio, sr = sf.read(audio_bytes_io)
        duration_seconds = librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted WAV file: {str(e)}")

    # if duration_seconds > 10:
    #     raise HTTPException(status_code=400, detail="Audio duration exceeds 10 seconds.")

    logger.info(f"Received file: {file.filename}, duration: {duration_seconds:.2f} seconds")

    # Save to a temp file safely for transcription
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run transcription off the main event loop
        transcription = await run_in_threadpool(transcribe_audio, tmp_path)

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Transcription failed.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content={"transcription": transcription})
