from datasets import load_dataset
from datasets import Audio
import tensorflow as tf
import librosa
import os
from utils import get_mel_spectrogram
from google.cloud import storage
from audio import log_mel_spectogram

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './secrets/data-service-account.json'


def load_data():
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    COLUMNS_TO_KEEP = ["sentence", "audio"]
    all_columns = dataset.column_names
    columns_to_remove = set(all_columns) - set(COLUMNS_TO_KEEP)
    dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def serialize_audio_with_transcript(mp3_filepath, transcript):
    """Serialize audio and transcript into TFRecord example"""

    mel_spect = log_mel_spectogram(mp3_filepath, n_mels=80, padding=1000)

    # Convert NumPy array to TensorFlow tensor
    mel_spect = tf.convert_to_tensor(mel_spect)

    # Serialize the tensor
    serialized_mel_spect = tf.io.serialize_tensor(mel_spect)

    # Serialize transcript
    serialized_transcript = tf.io.serialize_tensor(tf.constant(transcript))

    # Create a TFRecord example with audio and transcript
    feature = {
        'audio': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_mel_spect.numpy()])),
        'transcript': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_transcript.numpy()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


# Convert to TF-record
def create_tf_records(data, num_shards=200, prefix='', folder='data', bucket_name='common_voice_en_new', gcs_upload=True):
    num_records = len(data)
    step_size = num_records // num_shards + 1

    for i in range(65910, num_records, step_size):
        print("Creating shard:", (i // step_size), " from records:", i, "to", (i + step_size))
        path = '{}/{}_000{}.tfrecords'.format(folder, prefix, i // step_size)
        print(path)

        # Write the file
        with tf.io.TFRecordWriter(path) as writer:
            # Filter the subset of data to write to tfrecord file
            new_data = data[i:i + step_size]
            # Loop through the subset of data and write to tfrecord file
            for j in range(len(new_data['audio'])):
                # file not found error
                if not os.path.exists(new_data['audio'][j]['path']):
                    continue

                tf_example = serialize_audio_with_transcript(new_data['audio'][j]['path'], new_data['sentence'][j])
                writer.write(tf_example.SerializeToString())

        # Delete dataset
        del new_data

        # Upload to GCS
        if gcs_upload:
            upload_to_gcs(bucket_name, path, path.split('/')[-1])

        # Delete the file
        os.remove(path)


if __name__ == "__main__":
    dataset = load_data()
    create_tf_records(dataset, prefix='cv_13_0', folder='data', bucket_name='common_voice_en_new_3', gcs_upload=True)

