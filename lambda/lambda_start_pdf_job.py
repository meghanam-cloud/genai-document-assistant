import json
import boto3
import os
import time
import urllib.parse

s3 = boto3.client('s3')
textract = boto3.client('textract', region_name='ap-south-1')
ddb = boto3.client('dynamodb')

TABLE = os.environ.get('DDB_TABLE', 'textract_jobs')

def lambda_handler(event, context):
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    raw_key = record['s3']['object']['key']
    key = urllib.parse.unquote_plus(raw_key)
    print("Starting Textract job for:", key)

    # Start asynchronous Textract job
    resp = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    job_id = resp['JobId']
    now = int(time.time())

    # Write to DynamoDB
    ddb.put_item(
        TableName=TABLE,
        Item={
            'JobId': {'S': job_id},
            'Bucket': {'S': bucket},
            'Key': {'S': key},
            'Status': {'S': 'IN_PROGRESS'},
            'StartedAt': {'N': str(now)}
        }
    )

    print("Started Textract job:", job_id)
    return {'status': 'started', 'JobId': job_id}
