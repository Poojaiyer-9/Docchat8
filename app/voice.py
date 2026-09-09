import speech_recognition as sr
from gtts import gTTS
import os


def speech_to_text(audio_file_path: str = None) -> str:
    """Convert speech to text.

    If no audio file is provided, it listens to your microphone.
    Why? Lets users ask questions by speaking instead of typing.
    """
    recognizer = sr.Recognizer()

    if audio_file_path:
        # Use uploaded audio file
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
    else:
        # Use microphone
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

    # Convert speech to text using Google's free API (no key needed)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service unavailable"


def text_to_speech(text: str, output_path: str = "static/answer.mp3") -> str:
    """Convert text answer to speech audio.

    Why? Lets DocChat speak the answer back to the user, for hands-free use.
    """
    # Create gTTS object (Google Text-to-Speech, free)
    tts = gTTS(text=text, lang="en", slow=False)
    # Save to static folder so the web UI can play it
    os.makedirs("static", exist_ok=True)
    tts.save(output_path)
    return output_path  # Return path so we can send the audio file to the user
