
import tensorflow as tf
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './secrets/data-service-account.json'

# Replace with your GCP bucket path and TFRecord files
tfrecord_files = "gs://common_voice_en"

batch_size = 128


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    # Read TF Records
    feature_description = {
        'audio': tf.io.FixedLenFeature([], tf.string),
        'transcript': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    audio = tf.io.parse_tensor(parsed_example['audio'],
                               out_type=tf.float32)
    transcript = tf.io.parse_tensor(parsed_example['transcript'],
                                    out_type=tf.string)

    return audio, transcript


sample_dataset = tf.data.Dataset.list_files(tfrecord_files + '/*.tfrecords')
sample_dataset = sample_dataset.flat_map(tf.data.TFRecordDataset)
sample_dataset = sample_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
sample_dataset = sample_dataset.batch(batch_size)
sample_dataset = sample_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
print(sample_dataset)