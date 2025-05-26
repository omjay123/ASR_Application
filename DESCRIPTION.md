## Project Title: FastAPI-based ASR Application Using NVIDIA NeMo

---

## Features Successfully Implemented

### ASR Model Integration
- Successfully downloaded and used NVIDIA NeMo’s pretrained Hindi ASR model: `stt_hi_conformer_ctc_medium`.
- Converted the model to ONNX format using NeMo export utilities for optimized inference.
- Ensured compatibility with 16kHz mono audio clips of 5–10 seconds.

### FastAPI Server
- Developed a FastAPI application with a `POST /transcribe` endpoint.
- Integrated `ONNXRuntime` for fast, asynchronous-compatible inference.
- Validated uploaded `.wav` audio files for:
  - File format: `.wav`
  - Duration: 5–10 seconds
  - Sampling rate: 16kHz

### Containerization
- Created a `Dockerfile` using a lightweight Python slim base image.
- Installed minimal dependencies: FastAPI, Uvicorn, ONNXRuntime, NumPy, and NeMo export utilities.
- Configured Uvicorn to run on port `8000` inside the container.
- Optimized image size by excluding unnecessary development tools and cache.

### Testing and Sample Usage
- Verified API performance using `curl` and Postman with sample `.wav` files.
- Transcriptions returned as accurate JSON responses with low latency.

### Documentation
- Created a clear `README.md` with:
  - Instructions to build and run the container.
  - Example `curl`/Postman request.
  - Design considerations and usage tips.

---

## Issues Faced During Development

### Model Conversion to ONNX
- Faced compatibility issues with certain model layers during ONNX export.
- Required updating NeMo and tweaking export configurations.

### ONNX Inference Pipeline
- Initial errors due to dimension mismatches in audio features.
- Resolved by ensuring consistent audio preprocessing and input formatting.

### Asynchronous Inference
- `ONNXRuntime` doesn’t support async natively.
- Wrapped inference logic using `run_in_executor()` to maintain async FastAPI behavior.

### Container Dependencies
- Some NeMo and PyTorch dependencies bloated the image or conflicted with ONNX.
- Fixed by separating inference dependencies and using `onnxruntime-gpu` when needed.

---

## Components Not Fully Implemented & Limitations

### GPU Optimization in Docker
- ONNX model built for GPU, but GPU acceleration in Docker (`onnxruntime-gpu`) not fully tested due to hardware constraints.

### Optimizationn of the ASR Model
- Facing issues when connect the onnx inference for optimization. The code is attached.

### Language Limitation
- Only supports Hindi ASR (trained model).
- No multilingual or language detection functionality.

---

## Plan to Overcome Challenges

- **ONNX Optimization**: Use graph optimization passes and quantization for faster inference and smaller models.
- **GPU Acceleration**: Leverage `nvidia-docker` and `onnxruntime-gpu` on supported hardware.
- **Multilingual Support**: Introduce model selection or auto-detection for multilingual ASR.

---

## Known Limitations & Assumptions

- Input audio must be `.wav`, 16kHz, mono.
- No support for resampling or format conversion.
- Concurrency not benchmarked — single-instance inference tested only.
- Model only transcribes Hindi speech.
- No GPU acceleration in the current container deployment.

---

## Final Notes

This project delivers an end-to-end pipeline for Hindi ASR using NVIDIA NeMo, optimized with ONNX and served via FastAPI. The containerized architecture ensures reproducibility and provides a foundation for scalable, language-specific speech-to-text microservices.
