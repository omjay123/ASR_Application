from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import logger
from src.routes import transcribe

app = FastAPI(
    title="FastAPI-based ASR Application Using NVIDIA NeMo",
    description="FastAPI",
    version="1.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    logger.info("Fastapi running...")
    return {"message": "Hello World"}


app.include_router(transcribe.router)




