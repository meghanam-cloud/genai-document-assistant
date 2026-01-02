# qa_lambda.py
"""
QA Lambda — combined improvements:
- Adaptive + mode override
- Recruiter-friendly TL;DR + 3 bullets format for long mode
- Defaults: TOP_K=6, MAX_CONTEXT_CHARS=6000 (overridable via env)
- Dedupe sources before returning
- Title generation (from TL;DR)
- Robust Bedrock/Claude extraction (handles nested JSON)
- Requires: s3:GetObject, bedrock:InvokeModel
"""

import os
import json
import math
import re
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

cfg = Config(connect_timeout=10, read_timeout=120)
s3 = boto3.client("s3", config=cfg)
bedrock = boto3.client("bedrock-runtime",
                       region_name=os.environ.get("BEDROCK_REGION", "ap-south-1"),
                       config=cfg)

# ENV / tuning (defaults changed as requested)
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "meghana-doc-summarizer")
VECTOR_PREFIX = os.environ.get("VECTOR_PREFIX", "output/vectors/")
CHUNK_PREFIX = os.environ.get("CHUNK_PREFIX", "output/chunks/")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
CLAUDE_MODEL_ID = os.environ.get("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "bedrock-2023-05-31")
TOP_K = int(os.environ.get("TOP_K", "6"))                # default changed to 6
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "6000"))  # default changed to 6000

# ---------- utilities ----------
def debug(msg):
    print("[qa]", msg)

def invoke_model_payload(payload, model_id):
    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload).encode('utf-8')
    )
    return resp['body'].read().decode('utf-8', errors='ignore')

# ---------- embedding & retrieval ----------
def get_query_embedding(text):
    payload = {"inputText": text}
    body = invoke_model_payload(payload, EMBED_MODEL_ID)
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None
    emb = None
    if isinstance(parsed, dict):
        emb = parsed.get("embedding") or parsed.get("embeddings") or parsed.get("emb")
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
    elif isinstance(parsed, list):
        emb = parsed[0] if parsed and isinstance(parsed[0], list) else parsed
    return [float(x) for x in emb] if emb else None

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0

def list_vectors_for_doc(bucket, doc_id):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=VECTOR_PREFIX)
    keys = [o['Key'] for o in resp.get('Contents', []) if doc_id in o['Key']]
    if not keys:
        return None, "no vectors file found"
    keys.sort()
    key = keys[-1]
    obj = s3.get_object(Bucket=bucket, Key=key)
    vectors = json.loads(obj['Body'].read().decode('utf-8'))
    return vectors, key

def load_chunk_text(bucket, chunk_key):
    obj = s3.get_object(Bucket=bucket, Key=chunk_key)
    data = json.loads(obj['Body'].read().decode('utf-8'))
    return data.get("text", "")

# ---------- adaptive intent detection ----------
SHORT_Q = re.compile(r'\b(short|summary|summarize|tl;dr|brief|one[- ]line|in one|quick)\b', re.I)
LONG_Q = re.compile(r'\b(explain|explain in detail|detailed|why|how|describe|analysis|in depth|thorough|comprehensive)\b', re.I)

def detect_mode(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "medium"
    if LONG_Q.search(q):
        return "long"
    if SHORT_Q.search(q):
        return "short"
    return "medium"

# ---------- recruiter-friendly prompt settings (TL;DR + bullets) ----------
def prompt_settings_for_mode(mode: str):
    # recruiter-friendly instructions per mode
    if mode == "short":
        instr = "Provide a one-line TL;DR (one sentence) summarising the answer, then a 'Sources' line with chunk keys."
        tokens = 140
    elif mode == "long":
        instr = (
            "1) Start with a one-line TL;DR summary.\n"
            "2) Then provide exactly 3 short bullets (8–15 words each) capturing the top takeaways.\n"
            "3) Finally include a 'Sources' line listing chunk keys used.\n"
            "Write in a clear, recruiter-friendly tone."
        )
        tokens = 520
    else:
        instr = "Answer in 3–5 sentences, then include a 'Sources' line listing chunk keys used."
        tokens = 340
    return instr, tokens

def build_rag_prompt(question, excerpts, mode):
    instruction_text, _ = prompt_settings_for_mode(mode)
    system = (
        "You are a precise assistant. Use ONLY the provided document excerpts to answer the question. "
        "Do NOT invent facts or use outside knowledge. If the answer cannot be found wholly in the excerpts, reply exactly: \"I don't know.\""
    )
    style = f"{instruction_text} Cite sources in square brackets using the chunk_key (for example: [source: output/chunks/xyz/chunk_2.json])."
    ctx = ""
    for i, ex in enumerate(excerpts):
        ctx += f"Excerpt {i+1} (key: {ex['chunk_key']}):\n{ex['text']}\n\n"
    prompt = system + "\n\n" + style + "\n\n" + "Question: " + question + "\n\nDocument excerpts:\n\n" + ctx + "\n\nAnswer now:"
    return prompt

# ---------- robust Claude caller (extracts plain text) ----------
def call_claude(prompt, model_id, max_tokens=340, temperature=0.0):
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anthropic_version": ANTHROPIC_VERSION
    }

    body = None
    try:
        body = invoke_model_payload(payload, model_id)
    except ClientError as e:
        try:
            print("[qa][debug] ClientError.response:", json.dumps(getattr(e, "response", {}), ensure_ascii=False))
        except Exception:
            print("[qa][debug] ClientError:", str(e))
        raise

    def extract_text(obj):
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            # content list with text
            if 'content' in obj and isinstance(obj['content'], list):
                parts = []
                for c in obj['content']:
                    if isinstance(c, dict) and 'text' in c and isinstance(c['text'], str):
                        parts.append(c['text'])
                    elif isinstance(c, str):
                        parts.append(c)
                if parts:
                    return "\n\n".join(p.strip() for p in parts if p and p.strip())
            # outputs -> content
            if 'outputs' in obj and isinstance(obj['outputs'], list):
                for out in obj['outputs']:
                    if isinstance(out, dict):
                        c = out.get('content')
                        if isinstance(c, list) and c:
                            first = c[0]
                            if isinstance(first, dict) and 'text' in first:
                                return first['text'].strip()
                            if isinstance(first, str):
                                return first.strip()
                        if 'text' in out and isinstance(out['text'], str):
                            return out['text'].strip()
            # common keys
            for k in ('text', 'outputText', 'completion', 'generatedText', 'response'):
                if k in obj and isinstance(obj[k], str) and obj[k].strip():
                    return obj[k].strip()
            # fallback: walk values
            for v in obj.values():
                extracted = extract_text(v)
                if extracted:
                    return extracted
        if isinstance(obj, list):
            for item in obj:
                extracted = extract_text(item)
                if extracted:
                    return extracted
        return None

    # try parse
    parsed = None
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    # if parsed is a JSON string inside JSON, un-nest
    if isinstance(parsed, str):
        try:
            nested = json.loads(parsed)
            parsed = nested
        except Exception:
            pass

    if parsed is not None:
        text = extract_text(parsed)
        if text:
            return text

    # try to locate embedded JSON in raw body
    try:
        if isinstance(body, str):
            first_brace = body.find('{')
            if first_brace != -1:
                sub = body[first_brace:]
                try:
                    parsed2 = json.loads(sub)
                    text2 = extract_text(parsed2)
                    if text2:
                        return text2
                except Exception:
                    pass
    except Exception:
        pass

    return (body or "").strip()

# ---------- small title generator from TL;DR line ----------
def generate_title_from_tldr(tldr_line, max_words=7):
    if not tldr_line:
        return ""
    # take first sentence
    first = tldr_line.splitlines()[0].split('.')[0].strip()
    if not first:
        return ""
    words = [w.strip(" ,;:()\"'") for w in first.split() if w.strip()]
    # drop leading articles
    while words and words[0].lower() in ("the","a","an"):
        words = words[1:]
    chosen = words[:max_words]
    title = " ".join(chosen).title()
    return title if len(title.split()) >= 2 else ""

# ---------- handler ----------
def lambda_handler(event, context):

    # ✅ FIX: handle API Gateway JSON body
    if "body" in event and isinstance(event["body"], str):
        try:
            body = json.loads(event["body"])
        except Exception:
            body = {}
    else:
        body = event

    bucket = body.get("bucket", OUTPUT_BUCKET)
    doc_id = body.get("doc_id")
    question = body.get("question")
    top_k = int(body.get("top_k", TOP_K))
    mode_override = (body.get("mode") or "auto").lower()
    
    if not (doc_id and question):
        return {"status":"error","message":"missing doc_id or question"}

    # determine mode
    if mode_override in ("short","medium","long"):
        mode = mode_override
    else:
        mode = detect_mode(question)

    debug(f"mode={mode} (override={mode_override}) question='{question[:120]}'")

    # 1) embed query & load vectors
    q_emb = get_query_embedding(question)
    if not q_emb:
        return {"status":"error","message":"failed to embed query"}

    vectors, vkey_or_msg = list_vectors_for_doc(bucket, doc_id)
    if vectors is None:
        return {"status":"error","message": vkey_or_msg}

    scored = []
    for v in vectors:
        emb = v.get("embedding")
        if not emb:
            continue
        score = cosine(q_emb, emb)
        scored.append({"chunk_key": v.get("chunk_key"), "chunk_index": v.get("chunk_index"), "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]
    if not top:
        return {"status":"error","message":"no relevant chunks found"}

    # 2) fetch chunk texts and build excerpts (limit total chars)
    excerpts = []
    total = 0
    for t in top:
        try:
            txt = load_chunk_text(bucket, t["chunk_key"])
            if not txt:
                continue
            if total + len(txt) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total
                if remain <= 200:
                    break
                txt = txt[:remain]
            excerpts.append({"chunk_key": t["chunk_key"], "text": txt})
            total += len(txt)
        except Exception as e:
            debug(f"failed fetching chunk {t.get('chunk_key')}: {e}")
            continue

    if not excerpts:
        return {"status":"error","message":"no excerpt text available"}

    # 3) build prompt and call Claude
    prompt = build_rag_prompt(question, excerpts, mode)
    _, tokens = prompt_settings_for_mode(mode)

    try:
        answer = call_claude(prompt, CLAUDE_MODEL_ID, max_tokens=tokens, temperature=0.0)
    except Exception as e:
        debug(f"call_claude failed: {e}")
        return {"status":"error","message": str(e)}

    # 4) dedupe sources (preserve order) and align with excerpts
    # Build sources list from excerpts first (this matches cited items in prompt)
    sources = [e["chunk_key"] for e in excerpts]
    seen = set(); deduped = []
    for s in sources:
        if s not in seen:
            deduped.append(s); seen.add(s)
    sources = deduped

    # 5) generate short title from TL;DR if available
    # Try to extract the first line (TL;DR) from the model answer
    first_line = (answer.splitlines()[0].strip() if answer else "")
    title = generate_title_from_tldr(first_line, max_words=7)
    if not title:
        title = "Summary"

    return {"status":"success", "mode": mode, "title": title, "answer": answer, "sources": sources, "doc_id": doc_id}
