import streamlit as st
from audio_recorder_streamlit import audio_recorder
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import os

# Get a list of supported languages for the dropdown
langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)

def main():
    # Safety check for the image file
    if os.path.exists("Photo.jpg"):
        st.image("Photo.jpg")
    
    # Your updated title
    st.title("vikas b g")
    st.subheader("AI Audio Translation Hub")
    
    # Audio recorder component
    audio_bytes = audio_recorder(text="Click to record", neutral_color="#6aa36f")
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Language selection
        target_lang = st.selectbox("Select Target Language", list(langs_dict.keys()))
        target_code = langs_dict[target_lang]

        if st.button("Process & Translate"):
            try:
                with st.spinner("Processing audio..."):
                    # 1. Speech to Text
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        st.success(f"**Original:** {text}")

                    # 2. Translation
                    translated_text = GoogleTranslator(source='auto', target=target_code).translate(text)
                    st.info(f"**Translated ({target_lang}):** {translated_text}")

                    # 3. Text to Speech
                    tts = gTTS(text=translated_text, lang=target_code)
                    tts_fp = BytesIO()
                    tts.write_to_fp(tts_fp)
                    st.audio(tts_fp)
                    st.balloons()

            except Exception as e:
                st.error(f"Could not process audio: {e}")

if __name__ == "__main__":
    main()
