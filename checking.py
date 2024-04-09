import tensorflow as tf


tfrecord_files = 'data'

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

sample_dataset = tf.data.Dataset.list_files(tfrecord_files+'/*.tfrecords')
sample_dataset = sample_dataset.flat_map(tf.data.TFRecordDataset)
first_data = sample_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

count = 0
for sample in first_data:
  count+=1
print(count)