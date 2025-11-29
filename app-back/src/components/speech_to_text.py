
import speech_recognition as sr

def speech_to_text(audio_path):
    """
    Convert an audio file to text using SpeechRecognition (Google API).

    Args:
        audio_path: path to the audio file (WAV recommended)

    Returns:
        Recognized text string or None on failure.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="es-ES")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Recognition service error: {e}")
        return None
