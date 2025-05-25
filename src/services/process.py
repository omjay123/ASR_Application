import torch
import nemo.collections.asr as nemo_asr
from src.utils.logger import logger

# Load once at module import (avoid re-loading on every call)
try:
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_hi_conformer_ctc_medium")
    asr_model.eval()
    logger.info("ASR model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ASR model: {e}")
    asr_model = None

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a 16kHz mono WAV audio file to Hindi text.
    """
    if asr_model is None:
        raise RuntimeError("ASR model is not loaded")

    try:
        with torch.no_grad():
            transcription = asr_model.transcribe(paths2audio_files=[audio_path], batch_size=1, logprobs=False)
        return transcription[0]
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise
