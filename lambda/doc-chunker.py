# chunker_lambda.py
import os
import json
import re
import time
import boto3
from botocore.config import Config

cfg = Config(connect_timeout=10, read_timeout=60)
s3 = boto3.client('s3', config=cfg)

OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'meghana-doc-summarizer')
CHUNK_PREFIX = os.environ.get('CHUNK_PREFIX', 'output/chunks/')
MAX_CHARS = int(os.environ.get('MAX_CHUNK_CHARS', 3500))
OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 300))

def simple_sentence_split(text):
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def build_chunks(text):
    sents = simple_sentence_split(text)
    if not sents:
        return []
    chunks = []
    cur = ""
    for sent in sents:
        if len(cur) + len(sent) + 1 <= MAX_CHARS:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    out = []
    for i, ch in enumerate(chunks):
        if i > 0:
            prev = chunks[i - 1]
            overlap_text = prev[-OVERLAP:] if len(prev) > OVERLAP else prev
            out.append((overlap_text + " " + ch).strip())
        else:
            out.append(ch)
    return out

def lambda_handler(event, context):
    bucket = event.get("bucket", OUTPUT_BUCKET)
    key = event.get("key")

    if not key:
        return {"status": "error", "message": "missing key"}

    filename = os.path.basename(key)

    # 🔒 SAFETY: process only extracted text
    if not filename.endswith("_text.txt"):
        return {"status": "ignored", "reason": "not a text file"}

    # ✅ ONLY SOURCE OF TRUTH
    doc_id = filename.replace("_text.txt", "")

    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8", errors="ignore")

    chunks = build_chunks(text)
    if not chunks:
        return {"status": "error", "message": "no chunks"}

    ts = int(time.time())

    for i, c in enumerate(chunks):
        s3.put_object(
            Bucket=bucket,
            Key=f"{CHUNK_PREFIX}{doc_id}/chunk_{i}_{ts}.json",
            Body=json.dumps({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": c,
                "char_len": len(c)
            }).encode("utf-8"),
            ContentType="application/json"
        )

    return {"status": "success", "doc_id": doc_id, "count": len(chunks)}

   
 
