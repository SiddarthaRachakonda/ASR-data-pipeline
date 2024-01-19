ASR Data Preprocessing
======================
This repository contains the code for preprocessing the data for the Automatic Speech Recognition(ASR) project. The data is taken from the [Mozilla](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) dataset. The data is preprocessed in the following steps:
1. The audio files are stored in GCS bucket
2. The audio files are converted to flac format

---

**Starting the container:**
1. Building the docker image:
 
    ```bash
    docker build -t asr-data-preprocessing -f Dockerfile .
    ```
2. Running the docker container:
    ```bash
    docker run -it --rm --mount type=bind,source="$(pwd)",target=/app asr-data-preprocessing
    ```
---

**Running the code:**
Update hte `ENV` variables in the `Dockerfile`:
1. `GCS_BUCKET_NAME`: The GCS bucket name
2. `GOOGLE_APPLICATION_CREDENTIALS`: The path to the GCS credentials file

Upload the data to GCS bucket

    ```bash
    python data_preprocessing.py -u <audio_file>
    ```
Preprocess the data

    ```bash
    python data_preprocessing.py -p <audio_file>
    ```
Workflow:


