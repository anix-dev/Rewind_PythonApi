# app/db/s3_client.py
import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

# Load AWS environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # default region
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    raise Exception("❌ Missing AWS S3 environment variables")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def upload_file_to_s3(file_bytes: bytes, file_name: str, content_type: str) -> str | None:
    """
    Upload a file to S3 bucket
    :param file_bytes: File content in bytes
    :param file_name: Key/path in S3 bucket
    :param content_type: MIME type (e.g., 'image/jpeg')
    :return: Public URL of the uploaded file or None if failed
    """
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=file_name,
            Body=file_bytes,
            ContentType=content_type,
        )
        file_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_name}"
        return file_url
    except (NoCredentialsError, ClientError) as e:
        print(f"❌ S3 Upload Failed: {e}")
        return None


def delete_file_from_s3(file_name: str) -> bool:
    """
    Delete a file from S3 bucket
    :param file_name: Key/path in S3 bucket
    :return: True if deleted successfully, False otherwise
    """
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        print(f"✅ Deleted S3 file: {file_name}")
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"❌ S3 Deletion Failed for {file_name}: {e}")
        return False
