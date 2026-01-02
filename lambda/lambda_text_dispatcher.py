# dispatcher_lambda.py
import json
import boto3

lambda_client = boto3.client("lambda")

def extract_doc_id(key: str) -> str:
    # output/text/<doc_id>_text.txt
    filename = key.split("/")[-1]
    return filename.replace("_text.txt", "")

def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"[dispatcher] triggered for {bucket}/{key}")

    # 🔒 Safety: only process extracted text files
    if not key.endswith("_text.txt"):
        print("[dispatcher] not a _text.txt file, skipping")
        return {"status": "ignored"}

    doc_id = extract_doc_id(key)

    base_payload = {
        "bucket": bucket,
        "key": key
    }

    # 1️⃣ Summarizer (async)
    lambda_client.invoke(
        FunctionName="lambda_summarize_bedrock",
        InvocationType="Event",
        Payload=json.dumps(base_payload)
    )
    print("[dispatcher] summarizer invoked")

    # 2️⃣ Chunker (SYNC – must finish first)
    lambda_client.invoke(
        FunctionName="doc-chunker",
        InvocationType="RequestResponse",
        Payload=json.dumps(base_payload)
    )
    print("[dispatcher] chunker completed")

    # 3️⃣ Embedder (async – reads chunks)
    lambda_client.invoke(
        FunctionName="doc-embedder",
        InvocationType="Event",
        Payload=json.dumps({
            "bucket": bucket,
            "doc_id": doc_id
        })
    )
    print("[dispatcher] embedder invoked")

    return {
        "status": "ok",
        "doc_id": doc_id,
        "text_key": key
    }
