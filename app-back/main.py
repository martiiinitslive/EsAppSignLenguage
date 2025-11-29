from fastapi import FastAPI, UploadFile, File
import os
from src.components.audio_extractor import extract_audio_from_video
from src.components.speech_to_text import speech_to_text
# Future: model integration (TextToDictaModel)
# Future: images_to_video integration

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spanish sign-language interpretation API"}

# Endpoint to receive a video and process it
@app.post("/procesar_video/")
@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"temp_{file.filename}")
    # Save the uploaded video to a temporary file
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio
    audio_path = video_path + ".wav"
    extract_audio_from_video(video_path, audio_path)

    # Convert audio to text
    text = speech_to_text(audio_path)

    # Model inference and image/video generation would go here (future integration)
    # For now, return the recognized text only

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_path)

    # Keep original Spanish key for backward compatibility, and add English key
    return {"texto": text, "text": text}