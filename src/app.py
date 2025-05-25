import streamlit as st
import requests
from io import BytesIO
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi:8000/api")

st.title("Audio Transcription & Text-to-WAV")

tab1, tab2 = st.tabs(["Transcribe Audio", "Text to WAV"])

with tab1:
    st.header("Upload WAV file to transcribe")

    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    if uploaded_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
                try:
                    resp = requests.post(f"{BACKEND_URL}/transcribe", files=files)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("Transcription complete!")
                    st.text_area("Transcription", data.get("transcription", ""), height=150)
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("Convert text to WAV audio")

    input_text = st.text_area("Enter text to convert to WAV", height=100)
    if st.button("Convert to WAV"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Converting..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/text-to-wav/",
                        data={"text": input_text},
                        stream=True
                    )
                    resp.raise_for_status()
                    wav_bytes = BytesIO(resp.content)
                    st.audio(wav_bytes.read(), format="audio/wav")
                    st.download_button("Download WAV", data=resp.content, file_name="output.wav", mime="audio/wav")
                except Exception as e:
                    st.error(f"Error: {e}")
