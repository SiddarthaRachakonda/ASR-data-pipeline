import librosa
import numpy as np
# Allowed extensions for audio/video
ALLOWED_EXTENSIONS = ['mp3', 'wav', 'mp4', 'flac']


def check_validity(filename):
    """Check validity for audio and video files"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_gcs(file_path):
    """Check if the file path is a GCS path Starts with gs:// or https://storage.googleapis.com/"""
    return file_path.startswith("gs://") or file_path.startswith("https://storage.googleapis.com/")


def is_video(file_path):
    """Check if the file is a video file"""
    return file_path.endswith(".mp4")


def get_mel_spectrogram(y, sr, n_fft=2048, hop_length=1024):
    """Create mel spectrogram"""
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect
    