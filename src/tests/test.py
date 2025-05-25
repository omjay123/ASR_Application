import pytest
from httpx import AsyncClient
from fastapi import status
from src.main import app  # make sure you import your FastAPI instance

import os

@pytest.mark.asyncio
async def test_text_to_wav_success():
    payload = {"text": "Hello, this is a test audio"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/text-to-wav/", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

@pytest.mark.asyncio
async def test_text_to_wav_empty_text():
    payload = {"text": ""}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/text-to-wav/", json=payload)

    assert response.status_code == 200  # gTTS still generates audio even for empty string
    assert response.headers["content-type"] == "audio/wav"

@pytest.mark.asyncio
async def test_transcribe_success():
    test_wav_path = "tests/test_audio.wav"
    if not os.path.exists(test_wav_path):
        pytest.skip("Test WAV file not available")

    with open(test_wav_path, "rb") as f:
        files = {"file": ("test_audio.wav", f, "audio/wav")}
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/api/transcribe", files=files)

    assert response.status_code == 200
    assert "transcription" in response.json()

@pytest.mark.asyncio
async def test_transcribe_wrong_file_type():
    with open(__file__, "rb") as f:  # sending a .py file instead of .wav
        files = {"file": ("not_audio.txt", f, "text/plain")}
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/api/transcribe", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .wav files are supported."

@pytest.mark.asyncio
async def test_transcribe_invalid_audio():
    # Create fake invalid WAV bytes
    from io import BytesIO
    fake_wav = BytesIO(b"not real wav data")
    files = {"file": ("fake.wav", fake_wav, "audio/wav")}
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/transcribe", files=files)

    assert response.status_code == 400
    assert "Invalid or corrupted WAV file" in response.json()["detail"]
