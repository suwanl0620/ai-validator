import boto3
import os

def download_pdf_from_s3(bucket_name: str, key: str, local_path: str, profile_name: str = None):
    session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    s3 = session.client('s3')

    try:
        print(f"📥 Downloading from S3: bucket={bucket_name}, key={key} → {local_path}")
        s3.download_file(bucket_name, key, local_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Download failed: file not written to {local_path}")

        print(f"✅ File downloaded to {local_path}")

    except Exception as e:
        print(f"❌ S3 download failed: {str(e)}")
        raise
