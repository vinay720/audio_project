import streamlit as st
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import moviepy.editor as mp
import scipy.fftpack
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
        key="audio1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        audio = AudioSegment.from_file(path)

        start = st.number_input(
            "Start Time",
            0,
            key="start1"
        )

        end = st.number_input(
            "End Time",
            10,
            key="end1"
        )

        if st.button(
            "Trim Audio",
            key="trim_btn"
        ):

            trimmed = audio[start*1000:end*1000]

            output = "outputs/trimmed.wav"

            trimmed.export(output, format="wav")

            st.audio(output)

        convert_type = st.selectbox(
            "Convert To",
            ["mp3", "wav"],
            key="convert1"
        )

        if st.button(
            "Convert",
            key="convert_btn"
        ):

            output = f"outputs/converted.{convert_type}"

            audio.export(output, format=convert_type)

            st.audio(output)

        if st.button(
            "Normalize Audio",
            key="normalize_btn"
        ):

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
        type=["mp4"],
        key="video1"
    )

    if video:

        path = f"temp/{video.name}"

        with open(path, "wb") as f:
            f.write(video.read())

        st.video(path)

        clip = mp.VideoFileClip(path)

        if st.button(
            "Extract Audio",
            key="extract_btn"
        ):

            output = "outputs/audio.mp3"

            clip.audio.write_audiofile(output)

            st.audio(output)

        if st.button(
            "Resize 480p",
            key="resize_btn"
        ):

            resized = clip.resize(height=480)

            output = "outputs/resized.mp4"

            resized.write_videofile(output)

            st.video(output)

with tabs[2]:

    st.header("📊 Media Analyzer")

    file = st.file_uploader(
        "Upload Media",
        type=["mp3", "wav", "mp4"],
        key="media1"
    )

    if file:

        path = f"temp/{file.name}"

        with open(path, "wb") as f:
            f.write(file.read())

        st.write("📁 Filename:", file.name)

        st.write(
            "📦 File Size:",
            round(file.size / 1024, 2),
            "KB"
        )

        if "audio" in file.type:

            audio = AudioSegment.from_file(path)

            st.write(
                "⏱ Duration:",
                len(audio)/1000,
                "Seconds"
            )

            st.write(
                "🔊 Channels:",
                audio.channels
            )

            st.write(
                "🎚 Sample Rate:",
                audio.frame_rate,
                "Hz"
            )

            st.write(
                "💾 Bit Depth:",
                audio.sample_width * 8,
                "bits"
            )

            st.audio(path)

        elif "video" in file.type:

            clip = mp.VideoFileClip(path)

            st.write(
                "⏱ Duration:",
                round(clip.duration, 2),
                "Seconds"
            )

            st.write(
                "🎞 FPS:",
                clip.fps
            )

            st.write(
                "📺 Resolution:",
                clip.size
            )

            cap = cv2.VideoCapture(path)

            frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            bitrate = int(
                cap.get(cv2.CAP_PROP_BITRATE)
            )

            st.write(
                "🖼 Total Frames:",
                frame_count
            )

            st.write(
                "💾 Bitrate:",
                bitrate
            )

            cap.release()

            st.video(path)

with tabs[3]:

    st.header("🖼 Frame Processor")

    video = st.file_uploader(
        "Upload Video",
        type=["mp4"],
        key="frame1"
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

        with zipfile.ZipFile(
            zip_path,
            "w"
        ) as zipf:

            for file in os.listdir("frames"):

                zipf.write(
                    f"frames/{file}",
                    file
                )

        with open(zip_path, "rb") as f:

            st.download_button(
                "⬇ Download Frames ZIP",
                f,
                file_name="frames.zip",
                key="download_frames"
            )

with tabs[4]:

    st.header("📈 Audio Visualizer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"],
        key="visual1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        st.audio(path)

        y, sr_rate = librosa.load(
            path,
            sr=None
        )

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        librosa.display.waveshow(
            y,
            sr=sr_rate,
            ax=ax
        )

        ax.set_title("Waveform")

        st.pyplot(fig)

        D = librosa.stft(y)

        S_db = librosa.amplitude_to_db(
            np.abs(D),
            ref=np.max
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )

        img = librosa.display.specshow(
            S_db,
            sr=sr_rate,
            x_axis='time',
            y_axis='log',
            ax=ax2
        )

        fig2.colorbar(
            img,
            ax=ax2
        )

        st.pyplot(fig2)

with tabs[5]:

    st.header("🎵 Audio to WAV Converter")

    files = st.file_uploader(
        "Upload Audio Files",
        type=["mp3", "wav", "ogg", "flac"],
        accept_multiple_files=True,
        key="wav1"
    )

    if files:

        os.makedirs(
            "outputs/batch",
            exist_ok=True
        )

        for file in files:

            path = f"temp/{file.name}"

            with open(path, "wb") as f:
                f.write(file.read())

            audio = AudioSegment.from_file(path)

            output_name = (
                file.name.split(".")[0]
                + ".wav"
            )

            output_path = (
                f"outputs/batch/{output_name}"
            )

            audio.export(
                output_path,
                format="wav"
            )

        zip_path = "outputs/audio_wav.zip"

        with zipfile.ZipFile(
            zip_path,
            "w"
        ) as zipf:

            for root, dirs, filenames in os.walk(
                "outputs/batch"
            ):

                for filename in filenames:

                    file_path = os.path.join(
                        root,
                        filename
                    )

                    zipf.write(
                        file_path,
                        filename
                    )

        with open(zip_path, "rb") as f:

            st.download_button(
                "⬇ Download WAV ZIP",
                f,
                file_name="audio_wav.zip",
                key="wav_download"
            )

with tabs[6]:

    st.header("🎤 Voice Changer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"],
        key="voice1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        audio = AudioSegment.from_file(path)

        mode = st.selectbox(
            "Select Voice Effect",
            ["Robot", "Deep"],
            key="voice_mode"
        )

        if mode == "Robot":

            changed = audio.speedup(
                playback_speed=1.3
            )

        else:

            changed = audio.speedup(
                playback_speed=0.8
            )

        output = "outputs/voice.wav"

        changed.export(
            output,
            format="wav"
        )

        st.audio(output)

with tabs[7]:

    st.header("🎞 MP4 to GIF Converter")

    video = st.file_uploader(
        "Upload MP4",
        type=["mp4"],
        key="gif1"
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
        type=["wav"],
        key="speech1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        recognizer = sr.Recognizer()

        with sr.AudioFile(path) as source:

            audio_data = recognizer.record(source)

            text = recognizer.recognize_google(
                audio_data
            )

            st.text_area(
                "Recognized Text",
                text,
                height=300
            )

with tabs[9]:

    st.header("🥁 Beat Detection")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav"],
        key="beat1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr_rate = librosa.load(path)

        tempo, beats = librosa.beat.beat_track(
            y=y,
            sr=sr_rate
        )

        st.write("🎵 Tempo:", tempo)

        st.write(
            "🥁 Beat Frames:",
            beats
        )

with tabs[10]:

    st.header("📊 Spectrum Analyzer")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"],
        key="spectrum1"
    )

    if audio_file:

        path = f"temp/{audio_file.name}"

        with open(path, "wb") as f:
            f.write(audio_file.read())

        y, sr_rate = librosa.load(path)

        fft = np.abs(
            scipy.fftpack.fft(y)
        )

        freqs = scipy.fftpack.fftfreq(
            len(fft)
        ) * sr_rate

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            freqs[:5000],
            fft[:5000]
        )

        ax.set_title(
            "Audio Spectrum Analyzer"
        )

        st.pyplot(fig)
