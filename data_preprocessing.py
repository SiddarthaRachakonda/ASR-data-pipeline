# Video/Audio

# preference for both inference and training.

# make it executable.

import os
import argparse
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from google.cloud import storage
from utils import check_validity, is_gcs, is_video, get_mel_spectrogram


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    blob.make_public()
    return blob.public_url


def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    return audio_path


def download_from_gcs(file_path):
    """Download file from GCS"""
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(os.path.basename(file_path))
    return os.path.basename(file_path)


def train_pre_process(file_path, max_audio_size=10, sampling_rate=16000):
    """Pre-process the file for training"""

    y, sr = librosa.load(file_path, sr=sampling_rate)

    # Pad the audio file to 10 seconds
    if len(y) < max_audio_size * sampling_rate:
        y = np.pad(y, (0, max_audio_size * sampling_rate - len(y)), 'constant')
    else:
        y = y[:max_audio_size * sampling_rate]

    # transcript

    return get_mel_spectrogram(y, sr)



def inference_pre_process(file_path, max_audio_size=10, sampling_rate=16000):
    """Pre-process the file for inference"""

    y, sr = librosa.load(file_path, sr=sampling_rate)

    chunks = []
    mel_spects = []

    while len(y) > max_audio_size * sampling_rate:
        chunks.append(y[:max_audio_size * sampling_rate])
        mel_spects.append(get_mel_spectrogram(chunks[-1], sr))
        y = y[max_audio_size * sampling_rate:]

    if len(y) < max_audio_size * sampling_rate:
        y = np.pad(y, (0, max_audio_size * sampling_rate - len(y)), 'constant')
        mel_spects.append(get_mel_spectrogram(y, sr))
    return mel_spects


if __name__ == "__main__":
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument('-u', '--upload', type=str, help="Path to the file")
    parser.add_argument("-p", "--pre_process", type=str, help="Pre-process the file")
    parser.add_argument("-m", "--mode", type=str, help="Train or Inference mode either t or i")

    args = parser.parse_args()

    if args.upload:
        if not check_validity(args.upload):
            raise ValueError("Allowed file type are mp3, mp4, wav")
        gcs_link = upload_to_gcs(bucket_name, args.upload, os.path.basename(args.upload))
        print(gcs_link)

    if args.pre_process:
        file_path = args.pre_process
        if not check_validity(file_path):
            raise ValueError("Allowed file type are mp3, mp4, wav")

        mode = "t" if args.mode else "i"

        if is_gcs(file_path):
            file_path = download_from_gcs(file_path)

        if is_video(file_path):
            file_path = extract_audio(file_path)

        max_audio_size = 10
        sampling_rate = 16000

        if mode == "t":
            print(train_pre_process(file_path, max_audio_size, sampling_rate))
        else:
            print(inference_pre_process(file_path))
