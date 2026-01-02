import os
import boto3
import traceback
import time

textract = boto3.client('textract', region_name='ap-south-1')
ddb = boto3.client('dynamodb', region_name='ap-south-1')
s3 = boto3.client('s3', region_name='ap-south-1')

TABLE = os.environ.get('DDB_TABLE', 'textract_jobs')
OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'meghana-doc-summarizer')

def get_in_progress_jobs():
    resp = ddb.scan(
        TableName=TABLE,
        FilterExpression='#s = :inprogress',
        ExpressionAttributeNames={'#s': 'Status'},
        ExpressionAttributeValues={':inprogress': {'S': 'IN_PROGRESS'}}
    )
    return resp.get('Items', [])

def get_full_text_from_job(job_id):
    pages = []
    next_token = None
    while True:
        if next_token:
            resp = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
        else:
            resp = textract.get_document_text_detection(JobId=job_id)
        blocks = resp.get('Blocks', [])
        for b in blocks:
            if b.get('BlockType') == 'LINE':
                pages.append(b.get('Text'))
        next_token = resp.get('NextToken')
        if not next_token:
            break
    return "\n".join(pages)

def update_job_completed(job_id, out_key):
    now = int(time.time())
    ddb.update_item(
        TableName=TABLE,
        Key={'JobId': {'S': job_id}},
        UpdateExpression='SET #S = :s, CompletedAt = :c, OutputKey = :o',
        ExpressionAttributeNames={'#S': 'Status'},
        ExpressionAttributeValues={':s': {'S': 'COMPLETED'}, ':c': {'N': str(now)}, ':o': {'S': out_key}}
    )

def update_job_failed(job_id):
    ddb.update_item(
        TableName=TABLE,
        Key={'JobId': {'S': job_id}},
        UpdateExpression='SET #S = :s',
        ExpressionAttributeNames={'#S': 'Status'},
        ExpressionAttributeValues={':s': {'S': 'FAILED'}}
    )

def lambda_handler(event, context):
    try:
        items = get_in_progress_jobs()
        print("Found in-progress jobs:", len(items))
        for it in items:
            job_id = it['JobId']['S']
            bucket = it['Bucket']['S']
            key = it['Key']['S']
            print("Checking job:", job_id, "for", key)
            try:
                status_resp = textract.get_document_text_detection(JobId=job_id, MaxResults=1)
            except Exception as e:
                print("Error fetching job status for", job_id, e)
                continue
            job_status = status_resp.get('JobStatus')
            print("JobStatus for", job_id, "=", job_status)
            if job_status and job_status.upper() == 'SUCCEEDED':
                text = get_full_text_from_job(job_id)
                out_key = f"output/text/{job_id}_text.txt"
                s3.put_object(Bucket=OUTPUT_BUCKET, Key=out_key, Body=text.encode('utf-8'), ContentType='text/plain')
                update_job_completed(job_id, out_key)
                print("Saved output to", out_key)
            elif job_status and job_status.upper() in ('FAILED', 'PARTIAL_SUCCESS'):
                update_job_failed(job_id)
                print("Marked job failed:", job_id)
            else:
                print("Job not ready yet:", job_id)
        return {'status': 'ok', 'checked': len(items)}
    except Exception as e:
        print("Unhandled error in poller:", e)
        traceback.print_exc()
        raise
