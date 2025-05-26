import os
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from src.utils.logger import logger



MODEL_NAME = "stt_hi_conformer_ctc_medium"
ONNX_PATH = "stt_hi_conformer_ctc_medium.onnx"

# Load or export ONNX model and tokenizer
def load_model_and_tokenizer():
    try:
        if not os.path.exists(ONNX_PATH):
            logger.info("Exporting NeMo model to ONNX...")
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
            asr_model.export(ONNX_PATH)
            tokenizer = asr_model.tokenizer
            logger.info("ONNX export complete.")
        else:
            logger.info("ONNX model already exists.")
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
            tokenizer = asr_model.tokenizer

        ort_session = ort.InferenceSession(ONNX_PATH)
        logger.info("ONNX model loaded successfully.")
        return ort_session, tokenizer

    except Exception as e:
        logger.error(f"Failed to load or export model: {e}")
        raise


# Preprocess audio: mono, 16kHz, log-mel spectrogram with 80 features
def preprocess_audio(audio_path: str, target_sr=16000) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)

    if sr != target_sr:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to log-mel spectrogram (as expected by NeMo ONNX model)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80
    )(waveform)

    log_mel_spec = torch.log(mel_spec + 1e-10)
    return log_mel_spec.numpy().astype(np.float32)  # shape: (1, 80, T)


# Load model and tokenizer globally (so it's done only once)
ort_session, tokenizer = load_model_and_tokenizer()


# Transcribe audio file using ONNX model
def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        input_audio = preprocess_audio(audio_path)  # shape: (1, 80, T)
        length = np.array([input_audio.shape[-1]], dtype=np.int64)

        inputs = {
            'audio_signal': input_audio,
            'length': length
        }

        outputs = ort_session.run(None, inputs)
        logits = outputs[0]  # shape: (1, T, vocab_size)
        token_ids = np.argmax(logits, axis=-1).flatten().tolist()

        # Filter out-of-vocab token IDs
        valid_token_ids = [tid for tid in token_ids if 0 <= tid < tokenizer.vocab_size]

        if len(valid_token_ids) < len(token_ids):
            logger.warning(
                f"Invalid token IDs detected. Max ID: {max(token_ids)}, Vocab size: {tokenizer.vocab_size}"
            )

        transcript = tokenizer.ids_to_text(valid_token_ids)
        return transcript

    except Exception as e:
        logger.error(f"ONNX transcription failed: {e}")
        raise

# audio="demo.wav"
# print(transcribe_audio(audio))