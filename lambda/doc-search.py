import os, json, time, math
import boto3
from botocore.config import Config

cfg = Config(connect_timeout=10, read_timeout=120)
s3 = boto3.client("s3", config=cfg)
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION","ap-south-1"), config=cfg)

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET","meghana-doc-summarizer")
VECTOR_PREFIX = os.environ.get("VECTOR_PREFIX","output/vectors/")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID","amazon.titan-embed-text-v2:0")
TOP_K = int(os.environ.get("TOP_K","4"))

def invoke_model_payload(payload, model_id):
    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload).encode('utf-8')
    )
    return resp['body'].read().decode('utf-8', errors='ignore')

def get_query_embedding(text):
    payload = {"inputText": text}
    body = invoke_model_payload(payload, EMBED_MODEL_ID)
    parsed = json.loads(body) if body else None
    emb = None
    if isinstance(parsed, dict):
        emb = parsed.get("embedding") or parsed.get("embeddings") or parsed.get("emb")
        if emb and isinstance(emb, list) and isinstance(emb[0], list):
            emb = emb[0]
    elif isinstance(parsed, list):
        emb = parsed[0] if parsed and isinstance(parsed[0], list) else parsed
    return [float(x) for x in emb] if emb else None

def cosine(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0

def load_vectors_for_doc(bucket, doc_id):
    prefix = VECTOR_PREFIX
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [o['Key'] for o in resp.get('Contents',[]) if doc_id in o['Key']]
    if not keys:
        return None, "no vectors file found"
    keys.sort()
    key = keys[-1]
    obj = s3.get_object(Bucket=bucket, Key=key)
    vectors = json.loads(obj['Body'].read().decode('utf-8'))
    return vectors, key

def lambda_handler(event, context):
    """
    event:
    {
      "bucket":"meghana-doc-summarizer",
      "doc_id":"c39a7bb8300a582d5",
      "query":"How does social media affect career opportunities?",
      "top_k":4
    }
    """

    bucket = event.get("bucket", OUTPUT_BUCKET)
    doc_id = event.get("doc_id")
    query = event.get("query")
    top_k = int(event.get("top_k", TOP_K))

    if not (doc_id and query):
        return {"status":"error","message":"missing doc_id or query"}

    # embed query
    q_emb = get_query_embedding(query)
    if not q_emb:
        return {"status":"error","message":"failed to embed query"}

    # load vectors for this document
    vectors, key_or_msg = load_vectors_for_doc(bucket, doc_id)
    if vectors is None:
        return {"status":"error","message": key_or_msg}

    scores = []
    for v in vectors:
        emb = v.get("embedding")
        if not emb:
            continue
        score = cosine(q_emb, emb)
        scores.append({
            "chunk_key": v.get("chunk_key"),
            "chunk_index": v.get("chunk_index"),
            "score": float(score),
            "char_len": v.get("char_len")
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return {
        "status":"success",
        "doc_id": doc_id,
        "vectors_key": key_or_msg,
        "results": scores[:top_k]
    }
