import streamlit as st
import os
import zipfile
import uuid
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import moviepy.editor as mp
import scipy.fftpack
import speech_recognition as sr

from pydub import AudioSegment
from pydub.silence import detect_silence

# =========================
# SAFE IMPORT OPENCV
# =========================
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False
    st.warning("OpenCV not installed. Frame tools disabled.")

# =========================
# SESSION SAFE FOLDERS
# =========================
session_id = str(uuid.uuid4())[:8]

TEMP_DIR = f"temp/{session_id}"
OUT_DIR = f"outputs/{session_id}"
FRAME_DIR = f"frames/{session_id}"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="AI Media Utility Studio", layout="wide")
st.title("🎵🎥 AI Media Utility Studio")

tabs = st.tabs([
    "Audio Toolkit",
    "Video Toolkit",
    "Media Analyzer",
    "Frame Processor",
    "Audio Visualizer",
    "Audio to WAV",
    "Voice Changer",
    "MP4 to GIF",
    "Speech-to-Text",
    "Beat Detection",
    "Spectrum Analyzer"
])

# =========================
# TAB 1 - AUDIO TOOLKIT
# =========================
with tabs[0]:
    st.header("🎵 Audio Toolkit")

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        audio = AudioSegment.from_file(path)

        start = st.number_input("Start Time (sec)", 0)
        end = st.number_input("End Time (sec)", 10)

        if st.button("Trim Audio"):
            trimmed = audio[start*1000:end*1000]
            output = os.path.join(OUT_DIR, "trimmed.wav")
            trimmed.export(output, format="wav")
            st.audio(output)

        if st.button("Convert to WAV"):
            output = os.path.join(OUT_DIR, "converted.wav")
            audio.export(output, format="wav")
            st.audio(output)

        if st.button("Normalize"):
            normalized = audio.apply_gain(-audio.max_dBFS)
            output = os.path.join(OUT_DIR, "normalized.wav")
            normalized.export(output, format="wav")
            st.audio(output)

        st.write("Silence Detection:")
        silence = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)
        st.write(silence)

# =========================
# TAB 2 - VIDEO TOOLKIT
# =========================
with tabs[1]:
    st.header("🎥 Video Toolkit")

    video = st.file_uploader("Upload Video", type=["mp4"])

    if video:
        path = os.path.join(TEMP_DIR, video.name)

        with open(path, "wb") as f:
            f.write(video.read())

        st.video(path)

        try:
            clip = mp.VideoFileClip(path, audio=False)
        except Exception as e:
            st.error(f"Video load error: {e}")
            st.stop()

        if st.button("Extract Audio"):
            out = os.path.join(OUT_DIR, "audio.mp3")
            clip.audio.write_audiofile(out)
            st.audio(out)

        if st.button("Resize 480p"):
            resized = clip.resize(height=480)
            out = os.path.join(OUT_DIR, "resized.mp4")
            resized.write_videofile(out)
            st.video(out)

# =========================
# TAB 3 - MEDIA ANALYZER
# =========================
with tabs[2]:
    st.header("📊 Media Analyzer")

    file = st.file_uploader("Upload Media", type=["mp3", "wav", "mp4"])

    if file:
        path = os.path.join(TEMP_DIR, file.name)

        with open(path, "wb") as f:
            f.write(file.read())

        st.write("Filename:", file.name)
        st.write("Size (KB):", round(file.size / 1024, 2))

        if "audio" in file.type:
            audio = AudioSegment.from_file(path)

            st.write("Duration:", len(audio)/1000, "sec")
            st.write("Channels:", audio.channels)
            st.write("Sample Rate:", audio.frame_rate)
            st.write("Bit Depth:", audio.sample_width * 8)

            st.audio(path)

        elif "video" in file.type:
            try:
                clip = mp.VideoFileClip(path)
                st.write("Duration:", clip.duration)
                st.write("FPS:", clip.fps)
                st.write("Resolution:", clip.size)
            except:
                st.error("Video metadata error")

            if CV2_AVAILABLE:
                cap = cv2.VideoCapture(path)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.write("Total Frames:", frames)
                cap.release()

            st.video(path)

# =========================
# TAB 4 - FRAME PROCESSOR
# =========================
with tabs[3]:
    st.header("🖼 Frame Processor")

    if not CV2_AVAILABLE:
        st.warning("OpenCV required")
    else:
        video = st.file_uploader("Upload Video", type=["mp4"])

        if video:
            path = os.path.join(TEMP_DIR, video.name)

            with open(path, "wb") as f:
                f.write(video.read())

            cap = cv2.VideoCapture(path)

            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % 30 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)

                    cv2.imwrite(
                        os.path.join(FRAME_DIR, f"frame_{count}.jpg"),
                        edges
                    )

                count += 1

            cap.release()

            zip_path = os.path.join(OUT_DIR, "frames.zip")

            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file in os.listdir(FRAME_DIR):
                    zipf.write(os.path.join(FRAME_DIR, file), file)

            with open(zip_path, "rb") as f:
                st.download_button("Download Frames", f)

# =========================
# TAB 5 - AUDIO VISUALIZER
# =========================
with tabs[4]:
    st.header("📈 Audio Visualizer")

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        y, sr_rate = librosa.load(path, sr=None, mono=True)
        y = np.nan_to_num(y)

        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr_rate, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig2, ax2 = plt.subplots()

        img = librosa.display.specshow(
            S_db,
            sr=sr_rate,
            x_axis="time",
            y_axis="hz",
            hop_length=512,
            ax=ax2
        )

        fig2.colorbar(img, ax=ax2)
        ax2.set_title("Spectrogram")

        st.pyplot(fig2)

# =========================
# TAB 6 - AUDIO TO WAV
# =========================
with tabs[5]:
    st.header("🎵 Batch Converter")

    files = st.file_uploader(
        "Upload Audio Files",
        type=["mp3", "wav", "ogg"],
        accept_multiple_files=True
    )

    if files:
        batch_dir = os.path.join(OUT_DIR, "batch")
        os.makedirs(batch_dir, exist_ok=True)

        for file in files:
            path = os.path.join(TEMP_DIR, file.name)

            with open(path, "wb") as f:
                f.write(file.read())

            audio = AudioSegment.from_file(path)
            out = os.path.join(batch_dir, file.name.split(".")[0] + ".wav")
            audio.export(out, format="wav")

        zip_path = os.path.join(OUT_DIR, "batch.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for f in os.listdir(batch_dir):
                zipf.write(os.path.join(batch_dir, f), f)

        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", f)

# =========================
# TAB 7 - VOICE CHANGER
# =========================
with tabs[6]:
    st.header("🎤 Voice Changer")

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        audio = AudioSegment.from_file(path)

        mode = st.selectbox("Effect", ["Robot", "Deep"])

        if mode == "Robot":
            changed = audio.speedup(1.3)
        else:
            changed = audio.speedup(0.8)

        out = os.path.join(OUT_DIR, "voice.wav")
        changed.export(out, format="wav")

        st.audio(out)

# =========================
# TAB 8 - GIF
# =========================
with tabs[7]:
    st.header("🎞 MP4 to GIF")

    video = st.file_uploader("Upload MP4", type=["mp4"])

    if video:
        path = os.path.join(TEMP_DIR, video.name)

        with open(path, "wb") as f:
            f.write(video.read())

        clip = mp.VideoFileClip(path)
        gif = os.path.join(OUT_DIR, "out.gif")
        clip.write_gif(gif)

        st.image(gif)

# =========================
# TAB 9 - SPEECH TO TEXT
# =========================
with tabs[8]:
    st.header("🧠 Speech to Text")

    audio_file = st.file_uploader("Upload WAV", type=["wav"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
        except:
            text = "Could not recognize speech"

        st.text_area("Text", text)

# =========================
# TAB 10 - BEAT DETECTION
# =========================
with tabs[9]:
    st.header("🥁 Beat Detection")

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr_rate = librosa.load(path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr_rate)

        st.write("Tempo:", tempo)
        st.write("Beats:", beats)

# =========================
# TAB 11 - SPECTRUM
# =========================
with tabs[10]:
    st.header("📊 Spectrum Analyzer")

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if audio_file:
        path = os.path.join(TEMP_DIR, audio_file.name)

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr_rate = librosa.load(path)
        fft = np.abs(scipy.fftpack.fft(y))
        freqs = scipy.fftpack.fftfreq(len(fft)) * sr_rate

        fig, ax = plt.subplots()
        ax.plot(freqs[:5000], fft[:5000])
        ax.set_title("Spectrum")

        st.pyplot(fig)
