import boto3

def download_pdf_from_s3(bucket_name: str, key: str, local_path: str):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)

    # Testing that S3 document is being accessed
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read()
    print(content[:200])  # show first 200 bytes of the PDF
