# embedder_lambda.py
"""
Final Embedder Lambda (production-ready)
- Uses Amazon Titan embeddings via Bedrock Runtime
- Correct payload shape: {"inputText": "..."} for Titan v2
- PII redaction (emails, phone numbers, ORCID, URLs)
- Strict sanitization to avoid ValidationException
- Splits very large chunks into sub-chunks and averages embeddings
- Retries with exponential backoff and jitter
- Debug logging of Bedrock response on parse/validation failures
- Clean AWS-only (no local /mnt/data dependencies)
"""

import os
import json
import time
import random
import re
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ---------- CONFIG ----------
cfg = Config(connect_timeout=10, read_timeout=120)
s3 = boto3.client('s3', config=cfg)
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=os.environ.get('BEDROCK_REGION', 'ap-south-1'),
    config=cfg
)

OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'meghana-doc-summarizer')
CHUNK_PREFIX = os.environ.get('CHUNK_PREFIX', 'output/chunks/')
VECTOR_PREFIX = os.environ.get('VECTOR_PREFIX', 'output/vectors/')
EMBED_MODEL_ID = os.environ.get('EMBED_MODEL_ID', 'amazon.titan-embed-text-v2:0')

# Tunables (environment variables)
MAX_INPUT_CHARS = int(os.environ.get('MAX_INPUT_CHARS', '2000'))   # safe trimming length
SUB_CHUNK_CHARS = int(os.environ.get('SUB_CHUNK_CHARS', '1500'))   # when splitting very large chunks
RETRY_ATTEMPTS = int(os.environ.get('EMBED_RETRY_ATTEMPTS', '6'))
BASE_DELAY = float(os.environ.get('EMBED_BASE_DELAY', '0.5'))

# ---------- UTILITIES ----------
def debug(msg):
    print(f"[embedder] {msg}")

def redact_pii(text: str) -> str:
    """Redact common PII: emails, phones, ORCID, URLs."""
    if not isinstance(text, str):
        return ""
    # email
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    # phone numbers (loose match)
    text = re.sub(r'\b(?:\+?\d[\d\-\s]{5,}\d)\b', '[REDACTED_PHONE]', text)
    # ORCID (xxxx-xxxx-xxxx-xxxx)
    text = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[REDACTED_ORCID]', text)
    # URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[REDACTED_URL]', text)
    return text

def strict_clean_text(s: str) -> str:
    """
    Very strict sanitizer: normalize UTF-8, remove control chars and non-ASCII,
    allow conservative punctuation set, collapse whitespace, redact long digit runs.
    """
    if not isinstance(s, str):
        return ""
    # normalize and drop invalid bytes
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # remove control characters
    s = re.sub(r'[\x00-\x1F\x7F]', ' ', s)
    # strip non-ascii (removes emojis, many foreign scripts)
    s = re.sub(r'[^\x00-\x7F]', ' ', s)
    # allow alphanum, whitespace and conservative punctuation
    s = re.sub(r"[^A-Za-z0-9\s\.\,\;\:\!\?\@\#\%\&\(\)\-\+\/\'\"\$\-]", ' ', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # redact extremely long number sequences leftover
    s = re.sub(r'\b\d{10,}\b', '[REDACTED_NUMBER]', s)
    return s

def try_cut_at_sentence(text: str, max_chars: int) -> str:
    """Prefer cutting at a sentence boundary near max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind('. ')
    if last_period > max_chars * 0.5:
        return cut[:last_period+1]
    return cut

def split_into_subchunks(text: str, max_chars: int):
    """Split text into subchunks up to max_chars, preferring sentence boundaries."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = ""
    for sent in sents:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
        else:
            i = 0
            while i < len(c):
                out.append(c[i:i+max_chars])
                i += max_chars
    return out

def average_vectors(vectors):
    """Average list of numeric vectors (element-wise)."""
    if not vectors:
        return None
    length = len(vectors[0])
    for v in vectors:
        if len(v) != length:
            raise ValueError("Mismatched embedding lengths")
    avg = [0.0] * length
    for v in vectors:
        for i, val in enumerate(v):
            avg[i] += float(val)
    inv = 1.0 / len(vectors)
    return [x * inv for x in avg]

# ---------- Bedrock invoke wrapper ----------
def invoke_model_payload(payload, model_id):
    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload).encode('utf-8')
    )
    return resp['body'].read().decode('utf-8', errors='ignore')

def safe_invoke_embedding(raw_text, model_id):
    """
    Sanitize, redact, trim, then call Titan embeddings with the native payload
    {"inputText": "<text>"}.
    Retries on transient errors and logs Bedrock response when parsing fails.
    """
    # redact PII then strict sanitize
    text = redact_pii(raw_text or "")
    text = strict_clean_text(text)

    if not text:
        raise ValueError("Empty cleaned text")

    text_trim = try_cut_at_sentence(text, MAX_INPUT_CHARS)
    payload = {"inputText": text_trim}

    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            body = invoke_model_payload(payload, model_id)

            # Try parse response and extract embedding
            parsed = None
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = None

            emb = None
            if isinstance(parsed, dict):
                # common keys: "embedding" or "embeddings" or "emb"
                emb = parsed.get('embedding') or parsed.get('embeddings') or parsed.get('emb')
                # sometimes embeddings nested as [[...]]
                if emb and isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                    emb = emb[0]
            elif isinstance(parsed, list):
                # top-level vector or list-of-vectors
                if parsed and isinstance(parsed[0], (int, float)):
                    emb = parsed
                elif parsed and isinstance(parsed[0], list):
                    emb = parsed[0]

            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]

            # No embedding found — log raw body for debugging
            print("[embedder][debug] raw bedrock response (snippet):", repr(body)[:2000])
            raise RuntimeError("No embedding found in response")

        except ClientError as e:
            last_exc = e
            # Dump server error details for debugging
            try:
                err_info = getattr(e, "response", {})
                print("[embedder][debug] ClientError response Error:", json.dumps(err_info.get("Error", {}), ensure_ascii=False))
            except Exception:
                pass

            msg = str(e)
            if 'Throttl' in msg or 'Throttling' in msg:
                sleep = min(8.0, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 0.5)
                debug(f"Throttled, backing off {sleep:.2f}s")
                time.sleep(sleep)
                continue

            sleep = min(8.0, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 0.25)
            debug(f"ClientError: {msg[:200]}. Backing off {sleep:.2f}s")
            time.sleep(sleep)
            continue

        except Exception as e:
            last_exc = e
            msg = str(e)
            # If validation/schema issue, trim aggressively and retry
            if 'Validation' in msg or 'Malformed' in msg or 'schema' in msg.lower():
                debug(f"[WARN] Validation error: {msg[:200]}; trimming input and retrying")
                text_trim = try_cut_at_sentence(text_trim, max(300, int(len(text_trim) * 0.6)))
                payload = {"inputText": text_trim}
                time.sleep(0.25 + random.random() * 0.2)
                continue
            sleep = min(8.0, BASE_DELAY * (2 ** attempt)) + random.random() * 0.25
            time.sleep(sleep)
            continue

    raise last_exc or RuntimeError("Embedding failed without specific exception")

# ---------- S3 helpers ----------
def list_chunk_objects(bucket, doc_id):
    prefix = f"{CHUNK_PREFIX}{doc_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.json')]

def read_chunk(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj['Body'].read().decode('utf-8'))

# ---------- Lambda handler ----------
def lambda_handler(event, context):
    """
    event: { "bucket": "...", "doc_id": "..." }
    Returns: {"status":"success","vectors_key": "...", "count": N} or error
    """
    bucket = event.get('bucket', OUTPUT_BUCKET)
    doc_id = event.get('doc_id')

    if not doc_id:
        return {"status": "error", "message": "missing doc_id"}

    try:
        chunk_keys = list_chunk_objects(bucket, doc_id)
    except Exception as e:
        return {"status": "error", "message": f"failed to list chunk objects: {str(e)}"}

    if not chunk_keys:
        return {"status": "error", "message": "no chunk files found for doc_id"}

    vectors = []
    for key in chunk_keys:
        try:
            chunk = read_chunk(bucket, key)
            text = chunk.get('text', '') or ""
            if not text.strip():
                debug(f"Skipping empty chunk {key}")
                continue

            # If chunk is large, split into subchunks and average embeddings
            if len(text) > MAX_INPUT_CHARS:
                debug(f"Splitting large chunk {key} ({len(text)} chars)")
                subchunks = split_into_subchunks(text, SUB_CHUNK_CHARS)
                sub_embs = []
                for sc in subchunks:
                    try:
                        emb = safe_invoke_embedding(sc, EMBED_MODEL_ID)
                        if emb:
                            sub_embs.append(emb)
                    except Exception as e:
                        debug(f"Subchunk embed failed for {key}: {str(e)[:200]}")
                        continue
                if not sub_embs:
                    debug(f"No embeddings produced for subchunks of {key}")
                    continue
                final_emb = average_vectors(sub_embs)
            else:
                try:
                    final_emb = safe_invoke_embedding(text, EMBED_MODEL_ID)
                except Exception as e:
                    debug(f"Embedding failed for {key}: {str(e)[:200]}")
                    continue

            vectors.append({
                "chunk_key": key,
                "chunk_index": chunk.get('chunk_index'),
                "embedding": final_emb,
                "char_len": chunk.get('char_len')
            })

        except Exception as e:
            debug(f"Failed processing chunk {key}: {str(e)}")
            continue

    if not vectors:
        return {"status": "error", "message": "no embeddings produced"}

    ts = int(time.time())
    out_key = f"{VECTOR_PREFIX}{doc_id}_vectors_{ts}.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=out_key,
            Body=json.dumps(vectors, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
    except Exception as e:
        return {"status": "error", "message": f"failed to write vectors to s3: {str(e)}"}

    debug(f"Wrote {len(vectors)} vectors -> {out_key}")
    return {"status": "success", "vectors_key": out_key, "count": len(vectors)}
