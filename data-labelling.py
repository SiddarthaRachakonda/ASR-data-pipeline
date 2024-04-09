from datasets import load_dataset
from datasets import Audio
import pickle
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './secrets/data-service-account.json'


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    blob.make_public()
    return blob.public_url


import gcsfs

fs = gcsfs.GCSFileSystem(project='common-voice-13')



cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")
cv_13 = cv_13.cast_column("audio", Audio(sampling_rate=16000))





# Save first 100 data in local
cv_13 = cv_13.select(range(0, min(1000, len(cv_13))))
cv_13.save_to_disk("data")


# dataset_path = "/Users/Shared/ASR-data-pipeline"
# cv_13.save_to_cloud(dataset_path, fs)
#
# cv_13 = cv_13.cast_column("audio", Audio(sampling_rate=16000))
# count = 0
#
# transcripts = {}
# for sample in cv_13:
#     upload_to_gcs("asr-data-storage", sample['audio']['path'], os.path.basename(sample['audio']['path']))
#
#     transcripts[os.path.basename(sample['audio']['path'])] = sample['sentence']
#     if count == 10:
#         break
#     count += 1
#
# with open('transcripts.pkl', 'wb') as f:
#     pickle.dump(transcripts, f)

#upload_to_gcs("asr-data-storage", "cv_13_data_100/transcripts.pkl", "transcripts.pkl")

# for sample in cv_13:
#     print(sample)
#     data_iterator.append([sample['audio']['path'], sample['sentence'], sample['accent'], sample['audio']['array']])
#     if count == 10:
#         break
#
#     count += 1
#
# df = pd.DataFrame(data_iterator, columns=columns)
#
# from google.cloud import storage
# import os
#
# # Convert DataFrame to CSV
# csv_file = 'first-10-data.csv'
# df.to_csv(csv_file, index=False)
#
# # Set up Google Cloud Storage client
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../secrets/data-service-account.json'
# client = storage.Client()
# bucket = client.get_bucket('training-data-cv')
#
# # Upload the CSV file
# blob = bucket.blob(csv_file)
# blob.upload_from_filename(csv_file)
#
# # Optionally, delete the local CSV file after upload
# os.remove(csv_file)




# COLUMNS_TO_KEEP = ["sentence", "audio"]
# all_columns = cv_13.column_names
#
# columns_to_remove = set(all_columns) - set(COLUMNS_TO_KEEP)
#
# cv_13 = cv_13.remove_columns(columns_to_remove)
#
# # cv_13 = cv_13.map(lambda x: {"audio": x["audio"], "text": x["sentence"]}, batched=True, num_proc=4)
# subset = cv_13.select(range(0, min(200000, len(cv_13))))