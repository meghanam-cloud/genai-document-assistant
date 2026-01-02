# lambda_summary_final_full.py
import boto3
import os
import json
import time
import traceback
import datetime
import re
import random
from botocore.config import Config
from botocore.exceptions import ClientError

# ---------- Config ----------
cfg = Config(connect_timeout=10, read_timeout=50, retries={"max_attempts": 2})
s3 = boto3.client('s3', config=cfg)
bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'ap-south-1'), config=cfg)

OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET', 'meghana-doc-summarizer')
SUMMARY_PREFIX = os.environ.get('SUMMARY_PREFIX', 'output/summaries/')
MODEL_ID = os.environ.get('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
ANTHROPIC_VERSION = os.environ.get('ANTHROPIC_VERSION', 'bedrock-2023-05-31')
TITLE_OVERRIDE = os.environ.get('TITLE_OVERRIDE', '').strip()  # optional fixed override; leave empty for AI titles
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

# default uploaded_file_url (local path preserved per request)
DEFAULT_UPLOADED_FILE_URL = "/mnt/data/34fc9e21-492e-4f35-b325-b27fdc5a39c3.png"

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# ---------- Small logging helpers ----------
def mask_pii(text):
    if not text:
        return text
    t = text
    # mask emails
    t = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', t)
    # mask long digit runs (phones/IDs)
    t = re.sub(r'\b(?:\d[ -]*?){6,}\b', '[MASKED_NUM]', t)
    # mask long numeric sequences (credit card / ids)
    t = re.sub(r'\b(?:\d[ -]*?){12,19}\b', '[CREDIT_OR_ID]', t)
    return t

def redact_for_logs(s, max_chars=200):
    if not s:
        return ""
    s2 = s.replace("\n", " ").strip()
    s2 = mask_pii(s2)
    if len(s2) > max_chars:
        return s2[:max_chars] + "...[TRUNCATED]"
    return s2

def debug_log(msg, level="DEBUG"):
    lvl = level.upper()
    if LOG_LEVEL == "DEBUG":
        print(f"[{lvl}] {msg}")
    elif LOG_LEVEL == "INFO" and lvl in ("INFO", "WARN", "ERROR"):
        print(f"[{lvl}] {msg}")
    # otherwise suppress

# ---------- Robust extractor & sanitizers ----------
def extract_plain_text_from_raw(raw_body):
    if not raw_body:
        return ""
    try:
        parsed = json.loads(raw_body)
        texts = []
        def walk(o):
            if o is None:
                return
            if isinstance(o, str):
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and k.lower() == 'text' and isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                    else:
                        walk(v)
            elif isinstance(o, list):
                for item in o:
                    walk(item)
        walk(parsed)
        if texts:
            return "\n\n".join(t for t in texts if t).strip()
        if isinstance(parsed, dict):
            if 'results' in parsed and parsed['results']:
                r0 = parsed['results'][0]
                if isinstance(r0, dict):
                    for k in ('outputText','text','generatedText','completion'):
                        if k in r0 and isinstance(r0[k], str) and r0[k].strip():
                            return r0[k].strip()
            if 'outputs' in parsed and parsed['outputs']:
                o0 = parsed['outputs'][0]
                if isinstance(o0, dict):
                    c = o0.get('content')
                    if isinstance(c, list) and c:
                        first = c[0]
                        if isinstance(first, dict) and first.get('text'):
                            return first.get('text').strip()
                        if isinstance(first, str):
                            return first.strip()
                    if 'text' in o0 and isinstance(o0['text'], str) and o0['text'].strip():
                        return o0['text'].strip()
            for k in ('outputText','generatedText','text','completion','response'):
                if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                    return parsed[k].strip()
    except Exception:
        pass
    # regex fallback
    pattern = re.compile(r'(?i)["\']?text["\']?\s*:\s*["\']((?:\\.|[^"\'])*?)["\']', re.DOTALL)
    matches = pattern.findall(raw_body)
    if matches:
        def unescape(s):
            s = s.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\r', '\r')
            return s
        texts = [unescape(m).strip() for m in matches if unescape(m).strip()]
        if texts:
            return "\n\n".join(texts).strip()
    s = raw_body.strip()
    s = re.sub(r'^\s*Title:\s*\{.*?\}\s*', '', s, flags=re.DOTALL)
    s = re.sub(r'^\s*Title:\s*', '', s)
    lines = []
    for ln in s.splitlines():
        low = ln.lower()
        if any(x in low for x in ('stop_reason', '"id"', '"type"', '"role"', '"model"', '"usage"', 'input_tokens', 'output_tokens')):
            continue
        if re.match(r'^\s*[\{\}\[\]":,]+\s*$', ln):
            continue
        lines.append(ln)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r'^(here is( a| the)?(:)?\s*)', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^(here\'s( a| the)?(:)?\s*)', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned

def sanitize_model_text(raw_text):
    if not raw_text:
        return ""
    s = extract_plain_text_from_raw(raw_text)
    parts = [p.strip() for p in re.split(r'\n{2,}', s) if p.strip()]
    if not parts:
        return s.strip()
    paragraph = None
    for p in parts:
        if len(re.findall(r'\w+', p)) >= 5:
            paragraph = p
            break
    if paragraph is None:
        paragraph = parts[0]
    bullets = []
    for p in parts[1:]:
        for ln in p.splitlines():
            ln = ln.strip()
            if ln.startswith(('*', '-', '•')):
                bullets.append(ln.lstrip('*-• ').strip())
            else:
                if 3 <= len(ln.split()) <= 18 and not re.search(r'\"id\"|stop_reason|usage|output_tokens|input_tokens', ln.lower()):
                    bullets.append(ln)
    bullets = bullets[:3]
    final = paragraph
    if bullets:
        final += "\n\n" + "\n".join(f"• {b}" for b in bullets)
    return final.strip()

def softer_sanitize(raw_text):
    if not raw_text:
        return ""
    s = raw_text.strip()
    lines = s.splitlines()
    kept = []
    for ln in lines:
        low = ln.lower()
        if any(x in low for x in ('"id"', 'stop_reason', '"usage"', 'input_tokens', 'output_tokens')) and ':' in ln:
            continue
        if re.match(r'^\s*[\{\}\[\]":,]+\s*$', ln):
            continue
        kept.append(ln)
    s2 = "\n".join(kept).strip()
    s2 = re.sub(r'^\s*\{[^\n]*\}\s*', '', s2, count=1).strip()
    s2 = re.sub(r'^(here is( a| the)?(:)?\s*)', '', s2, flags=re.IGNORECASE).strip()
    s2 = re.sub(r'^(here\'s( a| the)?(:)?\s*)', '', s2, flags=re.IGNORECASE).strip()
    return s2

# ---------- Paragraph fixer (improves fluency, fixes fragments) ----------
def clean_and_fix_paragraph(paragraph):
    if not paragraph or not paragraph.strip():
        return paragraph
    p = re.sub(r'\s+', ' ', paragraph.strip())
    fragments = re.split(r'(?<=[.!?])\s+', p)
    i = 0
    out = []
    while i < len(fragments):
        frag = fragments[i].strip()
        if len(frag.split()) <= 4 and i + 1 < len(fragments):
            seq = [frag.rstrip('.?!').strip()]
            j = i + 1
            while j < len(fragments) and len(fragments[j].split()) <= 4:
                seq.append(fragments[j].rstrip('.?!').strip())
                j += 1
            if len(seq) >= 2:
                if len(seq) == 2:
                    joined = f"{seq[0]} and {seq[1]}"
                else:
                    joined = ", ".join(seq[:-1]) + " and " + seq[-1]
                out.append(joined)
                i = j
                continue
        out.append(frag.rstrip('.?!').strip())
        i += 1
    merged = []
    for s in out:
        if not s:
            continue
        if s and s[0].islower() and merged:
            merged[-1] = merged[-1].rstrip() + ", " + s
        else:
            merged.append(s)
    final_sents = []
    for s in merged:
        s = s.strip()
        if not s:
            continue
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        if not re.search(r'[.!?]$', s):
            s = s + '.'
        final_sents.append(s)
    paragraph_fixed = " ".join(final_sents).strip()
    paragraph_fixed = re.sub(r'\s+,', ',', paragraph_fixed)
    paragraph_fixed = re.sub(r'\s+\.', '.', paragraph_fixed)
    paragraph_fixed = re.sub(r'\s{2,}', ' ', paragraph_fixed)
    return paragraph_fixed

# ---------- Title heuristics ----------
def generate_title_from_paragraph(paragraph, max_words=7):
    if not paragraph:
        return ""
    first = re.split(r'[.!?]', paragraph)[0].strip()
    if not first:
        return ""
    words = [w.strip(" ,;:()\"'") for w in first.split() if w.strip()]
    while words and words[0].lower() in ("the", "a", "an"):
        words = words[1:]
    chosen = words[:max_words]
    title = " ".join(chosen).title()
    if len(title.split()) < 2:
        return ""
    return title

# ---------- Enforce paragraph + bullets (metadata) ----------
def enforce_paragraph_and_bullets(cleaned_text, title_candidate):
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        if len(ln.split()) <= 3 and not re.search(r'[.!?]$', ln):
            continue
        filtered.append(ln)
    joined = " ".join(filtered).strip()
    bullets = []
    paragraph = None
    if any(sym in cleaned_text for sym in ("\n• ", "\n* ", "\n- ")):
        parts = [p.strip() for p in re.split(r'\n{2,}', cleaned_text) if p.strip()]
        paragraph = parts[0] if parts else joined
        for ln in cleaned_text.splitlines():
            ln = ln.strip()
            if ln.startswith(("• ", "* ", "- ")):
                bullets.append(ln.lstrip("•*- ").strip())
    else:
        paragraph = joined
    sents = re.split(r'(?<=[.!?])\s+', paragraph)
    if len(sents) < 4:
        clauses = re.split(r',\s*', paragraph)
        paragraph = ". ".join([c.strip().rstrip(',.') for c in clauses[:5]]).strip()
        if not paragraph.endswith('.'):
            paragraph += '.'
    elif len(sents) > 8:
        paragraph = " ".join(sents[:6]).strip()
        if not paragraph.endswith('.'):
            paragraph += '.'
    paragraph = paragraph.strip()
    if paragraph and not paragraph[0].isupper():
        paragraph = paragraph[0].upper() + paragraph[1:]
    if paragraph and not paragraph.endswith('.'):
        paragraph += '.'
    if not bullets:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
        for s in sentences:
            words = s.split()
            cand = " ".join(words[:18]).rstrip(',.')
            if cand:
                bullets.append(cand)
            if len(bullets) >= 3:
                break
    clean_bullets = []
    for b in bullets:
        b = re.sub(r'\s+', ' ', b).strip().rstrip('.,;:')
        words = b.split()
        if len(words) < 2:
            continue
        if len(words) > 18:
            b = " ".join(words[:18])
        if not b[0].isupper():
            b = b[0].upper() + b[1:]
        clean_bullets.append(b)
        if len(clean_bullets) >= 3:
            break
    if len(clean_bullets) < 3:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
        for s in sents:
            cand = " ".join(s.split()[:18]).rstrip(',.')
            if cand and cand not in clean_bullets:
                clean_bullets.append(cand)
            if len(clean_bullets) >= 3:
                break
    clean_bullets = clean_bullets[:3]
    title = title_candidate or ""
    return paragraph, clean_bullets, title

# ---------- Sentence splitter ----------
def split_sentences_strict(text):
    if not text or not text.strip():
        return []
    t = re.sub(r'\s+', ' ', text.strip())
    sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(t) if s.strip()]
    return sents

# ---------- Bedrock invocation with retries & throttling-aware ----------
def invoke_model_payload(payload):
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload).encode('utf-8')
    )
    return resp['body'].read().decode('utf-8', errors='ignore')

def invoke_with_retries(payload, attempts=6, base_delay=0.5, max_delay=8.0):
    last_exc = None
    for i in range(attempts):
        try:
            return invoke_model_payload(payload)
        except ClientError as e:
            last_exc = e
            # inspect for throttling
            code = None
            try:
                code = e.response.get('Error', {}).get('Code')
            except Exception:
                pass
            throttled = False
            if code and 'Throttling' in code:
                throttled = True
            elif 'throttl' in str(e).lower():
                throttled = True
            if throttled:
                sleep = min(max_delay, base_delay * (2 ** i)) + random.uniform(0, 0.5)
                debug_log(f"InvokeModel throttled (attempt {i+1}/{attempts}). Backing off {sleep:.2f}s", level="WARN")
                time.sleep(sleep)
                continue
            else:
                sleep = min(max_delay, base_delay * (2 ** i)) + random.uniform(0, 0.2)
                debug_log(f"InvokeModel transient error (attempt {i+1}/{attempts}): {redact_for_logs(str(e))}. Backing off {sleep:.2f}s", level="WARN")
                time.sleep(sleep)
                continue
    raise last_exc

def invoke_anthropic_messages(user_msg, max_tokens=800, temperature=0.0):
    payload = {
        "messages": [
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": max_tokens,
        "anthropic_version": ANTHROPIC_VERSION,
        "temperature": temperature
    }
    return invoke_with_retries(payload, attempts=6, base_delay=0.5)

# ---------- Local fallback summarizer (deterministic) ----------
def local_fallback_summarizer(source_text, target_sentences=6):
    if not source_text or not source_text.strip():
        return "The document appears empty and cannot be summarized."
    sents = split_sentences_strict(source_text)
    if not sents:
        lines = [l.strip() for l in source_text.splitlines() if l.strip()]
        return " ".join(lines[:target_sentences])
    # select longest sentences as proxy for informative
    sents_sorted = sorted(sents, key=lambda s: len(s), reverse=True)
    chosen = sents_sorted[:target_sentences]
    chosen_set = set(chosen)
    ordered = [s for s in sents if s in chosen_set]
    if len(ordered) < 3:
        ordered = (ordered + sents)[:max(3, min(7, target_sentences))]
    return " ".join(ordered[:7])

# ---------- Title generator (Analytical & Balanced using model) ----------
def generate_title_with_model_analytical(paragraph, max_words=7):
    if not paragraph:
        return ""
    prompt = (
        "You are a concise editorial assistant. Create a short (3–7 words), analytical, balanced, recruiter-friendly title "
        "for the document summary provided below. Use a professional tone — examples: "
        "'The Dual Impact of Social Media on Modern Life', 'How Social Media Shapes Communication and Society'. "
        "Do NOT include 'Summary:' or 'Title:' prefixes. Provide ONLY the title on a single line.\n\n"
        "Summary paragraph:\n\n" + paragraph[:1500] + "\n\nTitle:"
    )
    try:
        raw = invoke_anthropic_messages(prompt, max_tokens=60, temperature=0.0)
        text = extract_plain_text_from_raw(raw)
        first_line = text.splitlines()[0].strip() if text else ""
        first_line = re.sub(r'[^\w\s-]$', '', first_line).strip()
        words = [w.strip(" ,;:()\"'") for w in first_line.split() if w.strip()]
        chosen = words[:max_words]
        title = " ".join(chosen).title()
        if len(title.split()) < 2:
            return first_line or ""
        return title
    except Exception:
        return ""

# ---------- Lambda handler ----------
def lambda_handler(event, context):
    try:
        # 🔹 Handle S3 trigger automatically
        if "Records" in event and event["Records"][0].get("eventSource") == "aws:s3":
            record = event["Records"][0]
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]
            uploaded_file_url = DEFAULT_UPLOADED_FILE_URL
        else:
            # 🔹 Manual test / API call
            bucket = event.get("bucket", OUTPUT_BUCKET)
            key = event.get("key")
            uploaded_file_url = event.get("uploaded_file_url") or DEFAULT_UPLOADED_FILE_URL

        if not bucket or not key:
            return {"status": "error", "message": "Missing bucket or key"}

        debug_log(f"input bucket/key: {bucket}/{key}", level="INFO")

        obj = s3.get_object(Bucket=bucket, Key=key)
        text = obj['Body'].read().decode('utf-8', errors='ignore')

        # --- Recruiter-ready strict prompt with style example (do not copy verbatim) ---
        style_example = (
            "Example style (do NOT copy verbatim):\n\n"
            "Social media has become a defining force in modern society, shaping how people communicate, learn, and build professional relationships. "
            "It enables instant global connectivity and provides access to information, opportunities, and communities. "
            "Platforms such as WhatsApp, Facebook, Twitter, and Instagram support networking, career growth, and creative expression. "
            "At the same time, they introduce challenges such as misinformation and privacy risks. "
            "Balancing these benefits and drawbacks is essential as digital engagement grows."
        )

        prompt_user = (
            "SYSTEM: You are a professional editor who writes polished, natural, and recruiter-friendly summaries.\n\n"
            "INSTRUCTIONS (YOU MUST FOLLOW):\n"
            "- Output exactly one single paragraph.\n"
            "- Write 5–7 fluent, natural sentences in plain text only.\n"
            "- No bullets, no lists, no headings, no metadata, no disclaimers.\n"
            "- Do NOT start with 'In summary' or 'Overall'.\n"
            "- Do NOT copy sentences verbatim from the example or the document; paraphrase the ideas smoothly.\n"
            "- Capture both positive and negative points if present in the document.\n"
            "- Keep tone clean, modern, and professional.\n"
            "- Redact specific personal details if needed.\n"
            "- Output ONLY the summary paragraph.\n\n"
            + style_example + "\n\n"
            "DOCUMENT:\n\n" + text[:8000]
        )

        start = time.time()
        used_local_fallback = False
        try:
            raw_body = invoke_anthropic_messages(prompt_user, max_tokens=700, temperature=0.0)
        except ClientError as e:
            debug_log("Model invoke failed after retries: " + redact_for_logs(str(e)), level="ERROR")
            fallback_summary = local_fallback_summarizer(text, target_sentences=6)
            raw_body = json.dumps({"text": fallback_summary})
            used_local_fallback = True
        except Exception as e:
            debug_log("Model invoke unexpected error: " + redact_for_logs(str(e)), level="ERROR")
            fallback_summary = local_fallback_summarizer(text, target_sentences=6)
            raw_body = json.dumps({"text": fallback_summary})
            used_local_fallback = True

        elapsed = time.time() - start

        debug_log("RAW_BODY_SNIPPET: " + redact_for_logs(raw_body), level="DEBUG")

        extracted = extract_plain_text_from_raw(raw_body)
        cleaned = sanitize_model_text(raw_body)

        debug_log("EXTRACTED_SNIPPET: " + redact_for_logs(extracted), level="DEBUG")
        debug_log("CLEANED_SNIPPET: " + redact_for_logs(cleaned), level="DEBUG")

        if not cleaned or len(cleaned.strip()) < 20:
            debug_log("WARN: cleaned empty/short — trying softer/extracted fallbacks", level="WARN")
            softer = softer_sanitize(raw_body if raw_body else extracted)
            if not softer or len(softer.strip()) < 20:
                softer = softer_sanitize(extracted)
            if softer and len(softer.strip()) >= 20:
                cleaned = softer
                debug_log(f"INFO: Using softer sanitized text (len={len(cleaned)})", level="INFO")
            else:
                cleaned = (extracted or raw_body or "").strip()
                if not cleaned:
                    cleaned = "The document contains text but the model response could not be parsed. Please check model output."

        # Candidate paragraph from cleaned text
        candidate_paragraph = cleaned.split("\n\n")[0].strip() if cleaned else ""
        candidate_paragraph = clean_and_fix_paragraph(candidate_paragraph)

        sents = split_sentences_strict(candidate_paragraph)

        # If sentence count not 5-7, retry once with stricter rewrite
        if len(sents) < 5 or len(sents) > 7:
            debug_log(f"WARN: sentence count {len(sents)} not in 5-7 — performing one retry", level="WARN")
            retry_prompt = (
                "You did not follow the rule. Rewrite the TEXT below into EXACTLY 5 to 7 complete sentences in ONE "
                "single paragraph. Do NOT include bullets, lists, headers, metadata, or any extra text. Keep language "
                "professional and concise.\n\nDOCUMENT:\n\n" + text[:8000]
            )
            try:
                raw_body2 = invoke_anthropic_messages(retry_prompt, max_tokens=700, temperature=0.0)
                extracted2 = extract_plain_text_from_raw(raw_body2)
                cleaned2 = sanitize_model_text(raw_body2) or extracted2 or raw_body2 or ""
                candidate_paragraph = cleaned2.split("\n\n")[0].strip() if cleaned2 else candidate_paragraph
                candidate_paragraph = clean_and_fix_paragraph(candidate_paragraph)
                sents = split_sentences_strict(candidate_paragraph)
                debug_log("DEBUG_RETRY_CLEANED_SNIPPET: " + redact_for_logs(candidate_paragraph), level="DEBUG")
            except Exception as e:
                debug_log("Retry for paragraph failed: " + redact_for_logs(str(e)), level="WARN")

        # Deterministic fallback expansion/trimming to ensure 5-7
        if len(sents) < 5:
            debug_log(f"WARN: still short ({len(sents)}). Applying deterministic expansion fallback.", level="WARN")
            clauses = [c.strip().rstrip(',.') for c in re.split(r',\s*', candidate_paragraph) if c.strip()]
            new_sents = []
            i = 0
            while i < len(clauses) and len(new_sents) < 5:
                part = clauses[i]
                if i + 1 < len(clauses) and len(part.split()) < 6:
                    part = part + ", " + clauses[i + 1]
                    i += 2
                else:
                    i += 1
                if not re.search(r'[.!?]$', part):
                    part = part.strip() + '.'
                new_sents.append(part.strip())
            while len(new_sents) < 5:
                if new_sents:
                    new_sents.append(new_sents[-1])
                else:
                    new_sents.append("The document provides information that cannot be summarized further.")
            sents = new_sents[:7]

        if len(sents) > 7:
            sents = sents[:7]

        final_paragraph = " ".join(s.strip().rstrip(' .') + '.' for s in sents)
        final_paragraph = re.sub(r'\s{2,}', ' ', final_paragraph).strip()

        # final sanitize for stray tokens
        if any(tok in final_paragraph for tok in ['{', '}', '[', ']', '•', '*\u200b']):
            final_paragraph = re.sub(r'[\{\}\[\]•\*]', '', final_paragraph)
            final_paragraph = re.sub(r'\s{2,}', ' ', final_paragraph).strip()

        paragraph = final_paragraph

        # ---------- Title generation with retry (Option A small patch) ----------
        title = ""
        # prefer explicit override if provided
        if TITLE_OVERRIDE:
            title = TITLE_OVERRIDE
        else:
            attempts = 3
            base_delay = 0.4
            for attempt in range(1, attempts + 1):
                try:
                    title_candidate = generate_title_with_model_analytical(paragraph, max_words=7)
                    if title_candidate and len(title_candidate.split()) >= 2:
                        title = title_candidate.strip()
                        debug_log(f"AI title generated on attempt {attempt}: '{redact_for_logs(title, max_chars=120)}'", level="INFO")
                        break
                    else:
                        debug_log(f"AI title attempt {attempt} returned empty/short result; retrying...", level="WARN")
                except Exception as e:
                    debug_log(f"AI title attempt {attempt} error: {redact_for_logs(str(e))}", level="WARN")
                if attempt < attempts:
                    sleep = min(2.0, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                    time.sleep(sleep)

        # Fallback to heuristic/title_override if AI didn't produce a title
        if not title:
            try:
                paragraph_for_metadata, bullets_for_metadata, _ = enforce_paragraph_and_bullets(cleaned, "")
                title_heuristic = generate_title_from_paragraph(paragraph_for_metadata, max_words=7)
                title = title_heuristic or (TITLE_OVERRIDE if TITLE_OVERRIDE else "Summary")
                debug_log(f"Using fallback heuristic/title_override for title: '{redact_for_logs(title, max_chars=120)}'", level="INFO")
            except Exception as e:
                title = TITLE_OVERRIDE if TITLE_OVERRIDE else "Summary"
                debug_log(f"Fallback title generation error; using default '{title}': {redact_for_logs(str(e))}", level="ERROR")

        title = title.strip()
        if title == "." or not title:
            title = TITLE_OVERRIDE if TITLE_OVERRIDE else "Summary"

        # Compose final summary text file content: Simple Title (single line), blank line, paragraph
        summary_txt_content = f"{title}\n\n{paragraph}"

        # metadata for internal use
        paragraph_for_metadata, bullets_for_metadata, _ = enforce_paragraph_and_bullets(cleaned, generate_title_from_paragraph(candidate_paragraph, max_words=7))
        paragraph_for_metadata = clean_and_fix_paragraph(paragraph_for_metadata)
        bullets_for_metadata = [b.rstrip(' .') + '.' for b in bullets_for_metadata]

        ts = int(time.time())
        base = os.path.basename(key).replace("_text.txt", "").replace(".txt", "")
        human_key = f"{SUMMARY_PREFIX}{base}_summary_{ts}.txt"
        meta_key = f"output/metadata/{base}_summary_{ts}.json"

        # Save summary txt (Title + blank line + paragraph)
        s3.put_object(Bucket=bucket, Key=human_key, Body=summary_txt_content.encode('utf-8'), ContentType='text/plain')

        # Save metadata JSON (includes title, original source key, uploaded file url)
        meta = {
            "job_id": base,
            "source_key": key,
            "uploaded_file_url": uploaded_file_url,
            "title": title,
            "summary": paragraph,
            "bullets": bullets_for_metadata,
            "model": MODEL_ID,
            "anthropic_version": ANTHROPIC_VERSION,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "summary_key": human_key,
            "used_local_fallback": bool(used_local_fallback)
        }
        s3.put_object(Bucket=bucket, Key=meta_key, Body=json.dumps(meta, ensure_ascii=False).encode("utf-8"), ContentType='application/json')

        debug_log(f"SUMMARY_OK job={base} duration_s={round(elapsed,2)} sentences={len(sents)} title='{redact_for_logs(title, max_chars=120)}' fallback={used_local_fallback}", level="INFO")

        return {"status": "success", "summary_key": human_key, "metadata_key": meta_key, "preview": paragraph[:900], "title": title, "duration_s": round(elapsed, 2)}

    except Exception as e:
        tb = traceback.format_exc()
        debug_log("ERROR in summarizer: " + redact_for_logs(str(e)), level="ERROR")
        debug_log("TRACE: " + redact_for_logs(tb, max_chars=1000), level="DEBUG")
        return {"status": "error", "error": str(e), "trace": tb}
