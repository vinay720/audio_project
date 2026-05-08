import streamlit as st
from audio_recorder_streamlit import audio_recorder
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import os

# 1. Setup Languages
langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)

def main():
    # 2. Display the Photo (Ensure Photo.JPG is in your GitHub repo)
    if os.path.exists("Photo.JPG"):
        st.image("Photo.JPG", width=300)
    
    st.title("GOWTHAM RJ")
    st.subheader("AI Audio Translation Hub")

    # 3. Audio Recording
    audio_bytes = audio_recorder(text="Click to record", neutral_color="#6aa36f")
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        target_lang = st.selectbox("Select Target Language", list(langs_dict.keys()))
        target_code = langs_dict[target_lang]

        if st.button("Process & Translate"):
            try:
                with st.spinner("Translating..."):
                    # Speech to Text
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                    
                    st.success(f"**Original:** {text}")

                    # Translation
                    translated_text = GoogleTranslator(source='auto', target=target_code).translate(text)
                    st.info(f"**Translated:** {translated_text}")

                    # Text to Speech
                    tts = gTTS(text=translated_text, lang=target_code)
                    tts_fp = BytesIO()
                    tts.write_to_fp(tts_fp)
                    st.audio(tts_fp)
                    
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
