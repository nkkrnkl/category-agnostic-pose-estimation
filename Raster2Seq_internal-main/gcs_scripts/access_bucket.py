from google.cloud import storage
import os

# Change if you want a different bucket, but this is what you specified
BUCKET_NAME = "dl-category-agnostic-pose-mp100-data"
LOCAL_DIR = "data"

def list_blobs():
    client = storage.Client()  # Uses application default credentials
    bucket = client.bucket(BUCKET_NAME)

    print(f"Listing objects in bucket: {BUCKET_NAME}")
    for blob in bucket.list_blobs():
        print(blob.name)

def download_blob(blob_name: str, destination_path: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(destination_path)
    print(f"Downloaded gs://{BUCKET_NAME}/{blob_name} -> {destination_path}")

def download_all_blobs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    for blob in bucket.list_blobs():
        local_path = os.path.join(LOCAL_DIR, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {blob.name} -> {local_path}")
        blob.download_to_filename(local_path)

if __name__ == "__main__":
    # Example: list all files
    # list_blobs()
    download_all_blobs()
    # Example: download one file (uncomment and change names as needed)
    # download_blob("path/in/bucket/file.ext", "local_file.ext")