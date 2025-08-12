
import speech_recognition as sr

def speech_to_text(audio_path):
    """
    Convierte un archivo de audio a texto usando SpeechRecognition y el motor de Google.
    audio_path: ruta al archivo de audio (WAV recomendado)
    return: texto reconocido o None si falla
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        texto = recognizer.recognize_google(audio, language="es-ES")
        return texto
    except sr.UnknownValueError:
        print("No se pudo entender el audio.")
        return None
    except sr.RequestError as e:
        print(f"Error al conectar con el servicio de reconocimiento: {e}")
        return None
