# FastAPI-based ASR Application Using NVIDIA NeMo

## Objective
This project implements and containerizes a FastAPI Python application that serves an Automatic Speech Recognition (ASR) model built with [NVIDIA NeMo](https://developer.nvidia.com/nvidia-nemo). The model is optimized for inference using **ONNX** or **TorchScript** and can transcribe audio clips of 5–10 seconds duration.

---

## Features
- ASR model based on NVIDIA NeMo.
- Model optimized using ONNX or TorchScript for fast inference.
- FastAPI REST API serving transcription requests.
- Dockerized for easy deployment.
- Supports uploading audio files via HTTP multipart/form-data.

---

## Requirements
- Docker & Docker Compose
- Python 3.10
- NVIDIA GPU (optional, for faster inference)
- CUDA drivers installed (if using GPU)

---

## Project Structure
```
├── src/ # Main source code folder
│ ├── configs/ # Configuration files
│ ├── models/ # Model loading and inference logic
│ ├── routes/ # FastAPI route definitions
│ ├── services/ # Business logic and utilities
│ ├── tests/ # Test scripts
│ ├── utils/ # Reusable code and core logic or process
│ ├── .env/ # env variables
│ ├── main.py/ # main application for fastapi
│ ├── app.py/ # streamlit app to integrate with backend for test
├── deploy.sh # Deployment helper script
├── docker-compose.yml # Docker compose file
├── Dockerfile.fastapi # Docker build instructions for fastapi
├── Dockerfile.streamlit # Docker build instructions for streamlit
├── README.md # This file
└── requirements.txt # Python dependencies list

```


---

## Setup and Installation (Local)

### Clone the repository and run the app

```bash
git clone https://github.com/omjay123/ASR_Application.git
cd ASR_Application
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000

```

# open it with browser
```bash
http://localhost:8000/api/docs
```

## Run the code using Docker

**Requirements**
- Docker Desktop

1. Make the `deploy.sh` script executable:

```bash
chmod +x deploy.sh
```

2. Run the deployment script:
```
bash deploy.sh
```


