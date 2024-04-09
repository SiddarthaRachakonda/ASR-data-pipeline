
from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './secrets/data-service-account.json'


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

if __name__ == "__main__":
    upload_to_gcs('common_voice_en_new_2', 'data/cv_13_0_00014.tfrecords', 'cv_13_0_00014.tfrecords')