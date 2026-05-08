import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import os

# 1. Setup Languages
# This gets a dictionary of supported languages for the dropdown
try:
    langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
except Exception:
    # Fallback in case of API connection issues
    langs_dict = {"english": "en", "hindi": "hi", "spanish": "es", "french": "fr"}

def main():
    # 2. Display the Photo
    # Ensure "Photo.JPG" is uploaded to the same folder as app.py on GitHub
    if os.path.exists("Photo.JPG"):
        st.image("Photo.JPG", width=300)
    
    st.title("GOWTHAM RJ")
    st.subheader("AI Audio Translation Hub")

    # 3. Audio Recording Component
    audio_bytes = audio_recorder(
        text="Click the mic to record",
        neutral_color="#6aa36f",
        icon_size="3x"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Language Selection
        target_lang = st.selectbox("Select Target Language", list(langs_dict.keys()))
        target_code = langs_dict[target_lang]

        if st.button("Process & Translate"):
            try:
                with st.spinner("Processing audio..."):
                    # Step A: Speech to Text
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                    
                    st.success(f"**Original Text:** {text}")

                    # Step B: Translation
                    translated_text = GoogleTranslator(source='auto', target=target_code).translate(text)
                    st.info(f"**Translated Text ({target_lang}):** {translated_text}")

                    # Step C: Text to Speech (Audio Output)
                    tts = gTTS(text=translated_text, lang=target_code)
                    tts_fp = BytesIO()
                    tts.write_to_fp(tts_fp)
                    st.audio(tts_fp)
                    
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please speak more clearly.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
