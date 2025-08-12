from fastapi import FastAPI, UploadFile, File
import os
from src.components.audio_extractor import extract_audio_from_video
from src.components.speech_to_text import speech_to_text
# from models.gen_model import TextToDictaModel  # Para el futuro: integración del modelo
# from src.components.images_to_video import images_to_video  # Para el futuro

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de interpretación de lengua de signos en español"}

# Endpoint para recibir un video y procesarlo
@app.post("/procesar_video/")
async def procesar_video(file: UploadFile = File(...)):
    # Guardar el archivo de video temporalmente
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extraer audio
    audio_path = video_path + ".wav"
    extract_audio_from_video(video_path, audio_path)

    # Convertir audio a texto
    texto = speech_to_text(audio_path)

    # Aquí iría la llamada al modelo y la generación de imágenes y video
    # Por ahora, solo devolvemos el texto reconocido

    # Limpiar archivos temporales
    os.remove(video_path)
    os.remove(audio_path)

    return {"texto": texto}