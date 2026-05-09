import streamlit as st
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import moviepy.editor as mp
import scipy.fftpack
import soundfile as sf
import speech_recognition as sr

from pydub import AudioSegment
from pydub.silence import detect_silence

try:
    import cv2
except:
    st.error("OpenCV not installed")

os.makedirs("temp", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("frames", exist_ok=True)

st.set_page_config(
    page_title="AI Media Utility Studio",
    layout="wide"
)

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

with tabs[0]:

    st.header("🎵 Audio Toolkit")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"],
        key="audio_toolkit"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        audio = AudioSegment.from_file(path)

        start = st.number_input("Start Time", 0)
        end = st.number_input("End Time", 10)

        if st.button("Trim Audio"):

            trimmed = audio[start*1000:end*1000]

            output = "outputs/trimmed.wav"

            trimmed.export(output, format="wav")

            st.audio(output)

        convert_type = st.selectbox(
            "Convert To",
            ["mp3", "wav"]
        )

        if st.button("Convert"):

            output = f"outputs/converted.{convert_type}"

            audio.export(output, format=convert_type)

            st.audio(output)

        if st.button("Normalize Audio"):

            normalized = audio.normalize()

            output = "outputs/normalized.wav"

            normalized.export(output, format="wav")

            st.audio(output)

        silence = detect_silence(
            audio,
            min_silence_len=1000,
            silence_thresh=-40
        )

        st.write("Silence Parts:", silence)

with tabs[1]:

    st.header("🎥 Video Toolkit")

    video = st.file_uploader(
        "Upload Video",
        type=["mp4"]
    )

    if video:

        path = f"temp/{video.name}"

        with open(path, "wb") as f:
            f.write(video.read())

        st.video(path)

        clip = mp.VideoFileClip(path)

        if st.button("Extract Audio"):

            output = "outputs/audio.mp3"

            clip.audio.write_audiofile(output)

            st.audio(output)

        if st.button("Resize 480p"):

            resized = clip.resize(height=480)

            output = "outputs/resized.mp4"

            resized.write_videofile(output)

            st.video(output)

with tabs[2]:

    st.header("📊 Media Analyzer")

    file = st.file_uploader(
        "Upload Media",
        type=["mp3", "wav", "mp4"]
    )

    if file:

        path = f"temp/{file.name}"

        with open(path, "wb") as f:
            f.write(file.read())

        st.write("📁 Filename:", file.name)
        st.write("📦 File Size:", round(file.size / 1024, 2), "KB")

        if "audio" in file.type:

            audio = AudioSegment.from_file(path)

            st.write("⏱ Duration:", len(audio)/1000, "Seconds")
            st.write("🔊 Channels:", audio.channels)
            st.write("🎚 Sample Rate:", audio.frame_rate, "Hz")
            st.write("💾 Bit Depth:", audio.sample_width * 8, "bits")

            st.audio(path)

        elif "video" in file.type:

            clip = mp.VideoFileClip(path)

            st.write("⏱ Duration:", round(clip.duration, 2), "Seconds")
            st.write("🎞 FPS:", clip.fps)
            st.write("📺 Resolution:", clip.size)

            cap = cv2.VideoCapture(path)

            frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            bitrate = int(
                cap.get(cv2.CAP_PROP_BITRATE)
            )

            st.write("🖼 Total Frames:", frame_count)
            st.write("💾 Bitrate:", bitrate)

            cap.release()

            st.video(path)

with tabs[3]:

    st.header("🖼 Frame Processor")

    video = st.file_uploader(
        "Upload Video",
        type=["mp4"]
    )

    if video:

        path = f"temp/{video.name}"

        with open(path, "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture(path)

        count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if count % 30 == 0:

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                edges = cv2.Canny(
                    gray,
                    100,
                    200
                )

                cv2.imwrite(
                    f"frames/frame_{count}.jpg",
                    edges
                )

            count += 1

        cap.release()

        zip_path = "outputs/frames.zip"

        with zipfile.ZipFile(zip_path, "w") as zipf:

            for file in os.listdir("frames"):

                zipf.write(
                    f"frames/{file}",
                    file
                )

        with open(zip_path, "rb") as f:

            st.download_button(
                "⬇ Download Frames ZIP",
                f,
                file_name="frames.zip"
            )

with tabs[4]:

    st.header("📈 Audio Visualizer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"]
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        y, sr = librosa.load(path, sr=None)

        fig, ax = plt.subplots(figsize=(12, 4))

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        ax.set_title("Waveform")

        st.pyplot(fig)

        D = librosa.stft(y)

        S_db = librosa.amplitude_to_db(
            np.abs(D),
            ref=np.max
        )

        fig2, ax2 = plt.subplots(figsize=(12, 4))

        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='log',
            ax=ax2
        )

        fig2.colorbar(img, ax=ax2)

        st.pyplot(fig2)

with tabs[5]:

    st.header("🎵 Audio to WAV Converter")

    files = st.file_uploader(
        "Upload Audio Files",
        type=["mp3", "wav", "ogg", "flac"],
        accept_multiple_files=True
    )

    if files:

        os.makedirs("outputs/batch", exist_ok=True)

        for file in files:

            path = f"temp/{file.name}"

            with open(path, "wb") as f:
                f.write(file.read())

            audio = AudioSegment.from_file(path)

            output_name = file.name.split(".")[0] + ".wav"

            output_path = f"outputs/batch/{output_name}"

            audio.export(output_path, format="wav")

        zip_path = "outputs/audio_wav_files.zip"

        with zipfile.ZipFile(zip_path, "w") as zipf:

            for root, dirs, filenames in os.walk("outputs/batch"):

                for filename in filenames:

                    file_path = os.path.join(root, filename)

                    zipf.write(file_path, filename)

        with open(zip_path, "rb") as f:

            st.download_button(
                "⬇ Download WAV ZIP",
                f,
                file_name="audio_wav_files.zip"
            )

with tabs[6]:

    st.header("🎤 Voice Changer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"]
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        audio = AudioSegment.from_file(path)

        mode = st.selectbox(
            "Select Voice Effect",
            ["Robot", "Deep"]
        )

        if mode == "Robot":

            changed = audio.speedup(
                playback_speed=1.3
            )

        else:

            changed = audio.speedup(
                playback_speed=0.8
            )

        output = "outputs/voice_changed.wav"

        changed.export(output, format="wav")

        st.audio(output)

with tabs[7]:

    st.header("🎞 MP4 to GIF Converter")

    video = st.file_uploader(
        "Upload MP4",
        type=["mp4"]
    )

    if video:

        path = f"temp/{video.name}"

        with open(path, "wb") as f:
            f.write(video.read())

        clip = mp.VideoFileClip(path)

        gif_path = "outputs/output.gif"

        clip.write_gif(gif_path)

        st.image(gif_path)

with tabs[8]:

    st.header("🧠 Speech-to-Text")

    audio_file = st.file_uploader(
        "Upload WAV File",
        type=["wav"]
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        r = sr.Recognizer()

        with sr.AudioFile(path) as source:

            audio_data = r.record(source)

            text = r.recognize_google(audio_data)

            st.text_area(
                "Recognized Text",
                text,
                height=300
            )

with tabs[9]:

    st.header("🥁 Beat Detection")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"]
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr = librosa.load(path)

        tempo, beats = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        st.write("🎵 Tempo:", tempo)
        st.write("🥁 Beat Frames:", beats)

with tabs[10]:

    st.header("📊 Spectrum Analyzer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"]
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr = librosa.load(path)

        fft = np.abs(
            scipy.fftpack.fft(y)
        )

        freqs = scipy.fftpack.fftfreq(
            len(fft)
        ) * sr

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            freqs[:5000],
            fft[:5000]
        )

        ax.set_title("Audio Spectrum Analyzer")

        st.pyplot(fig)
