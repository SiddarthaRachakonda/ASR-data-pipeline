import librosa
import numpy as np
import tensorflow as tf
from subprocess import CalledProcessError, run

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 10


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def log_mel_spectogram(audio, n_mels=80, padding=0):
    """Create log mel spectrogram from audio"""

    # Load audio
    if not isinstance(audio, tf.Tensor):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = tf.convert_to_tensor(audio)

    if padding > 0:
        audio = tf.pad(audio, [[0, padding]])

    # STFT
    stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT)

    stft = tf.transpose(stft)

    magnitudes = tf.abs(stft[..., :-1]) ** 2

    # Mel filter
    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)

    # Mel spectrogram
    mel_spect = filters @ magnitudes

    # Logarithm
    log_spect = tf.math.log(tf.maximum(mel_spect, 1e-10)) / tf.math.log(10.0)
    log_spect = tf.maximum(log_spect, tf.reduce_max(log_spect) - 8.0)
    log_spect = (log_spect + 4.0) / 4.0

    return log_spect
