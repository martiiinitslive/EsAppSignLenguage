"""
Módulo para extraer el audio de un archivo de video.
"""

from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extrae el audio de un archivo de video y lo guarda en output_audio_path.
    video_path: ruta al archivo de video de entrada
    output_audio_path: ruta donde se guardará el archivo de audio extraído
    """
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(output_audio_path)
        else:
            raise ValueError("El video no contiene pista de audio.")
