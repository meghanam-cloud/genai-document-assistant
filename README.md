# GenAI Document Assistant (Serverless RAG on AWS)

A fully serverless **GenAI-powered document assistant** that allows users to upload PDFs,
automatically extract and summarize content, and ask natural-language questions using
Retrieval-Augmented Generation (RAG).


## What This Project Does

- Upload a PDF → text is extracted automatically
- Document is summarized using Generative AI
- Text is chunked and embedded into vectors
- Users can ask questions and receive grounded answers
- Answers include **source references** (no hallucinations)

## Key Technologies

- **Amazon S3** – Document storage & vector store
- **AWS Lambda** – Serverless processing
- **Amazon Textract** – OCR (async)
- **Amazon Bedrock (Claude + Titan)** – GenAI & embeddings
- **API Gateway** – Q&A endpoint
- **HTML + JavaScript** – Lightweight UI

## Architecture Overview

The system consists of two independent pipelines:

### 1️⃣ Automatic Document Processing
- S3 upload triggers Textract
- Extracted text is summarized, chunked, and embedded
- Vectors are stored in S3 (cost-optimized design)

### 2️⃣ User-driven Q&A (RAG)
- UI sends questions via API Gateway
- Relevant chunks are retrieved using vector similarity
- Claude answers strictly from retrieved context

## Repository Structure
genai-document-assistant/
├── architecture/
│   ├── architecture.png        # Full system diagram (draw.io export)
│   └── architecture.md         # Written explanation of the diagram
│
├── frontend/
│   └── index.html              # UI (HTML + JS)
│
├── lambda/
│   ├── lambda_start_pdf_job.py
│   ├── lambda_poll_textract.py
│   ├── lambda_text_dispatcher.py
│   ├── lambda_summarize_bedrock.py
│   ├── doc-chunker.py
│   ├── doc-embedder.py
│   ├── doc-search.py
│   └── qa_lambda.py
│
├── screenshots/
│   ├── s3-upload.png
│   ├── s3-output.png
│   ├── ui-input.png
│   └── ui-answer.png
│
└── README.md
📌 Full architecture details are available in: 'architecture/architecture.md'


## Demo Screenshots

The following screenshots demonstrate the complete end-to-end flow of the
GenAI Document Assistant, from document upload to grounded question answering.

### 1️⃣ Document Upload
- **File:** `screenshots/s3-upload.png`
- Shows a PDF uploaded to the S3 `input/pdfs/` bucket
- This upload automatically triggers the processing pipeline

### 2️⃣ Extracted Text Output
- **File:** `screenshots/s3-output.text.png`
- Displays the raw text extracted by Amazon Textract
- Stored as the single source of truth for downstream processing

### 3️⃣ Generated Document Summary
- **File:** `screenshots/s3-output.summaries.png`
- Shows the AI-generated document summary created using Amazon Bedrock (Claude)

### 4️⃣ Chunked Document Segments
- **File:** `screenshots/s3-output.chunks.png`
- Displays semantic text chunks used for retrieval
- Each chunk preserves context with overlap

### 5️⃣ Vector Embeddings Store
- **File:** `screenshots/s3-output.vectors.png`
- Shows vector embeddings stored in S3
- Demonstrates a cost-optimized alternative to OpenSearch

### 6️⃣ Q&A UI – User Input
- **File:** `screenshots/ui-input.png`
- Shows the web UI where a user enters:
  - API Gateway URL
  - Document ID
  - Natural-language question

### 7️⃣ Q&A UI – Grounded Answer
- **File:** `screenshots/ui-answer.png`
- Displays the AI-generated answer
- Includes source chunk references to prevent hallucinations


## Why This Project Matters

This project demonstrates how to build a production-style GenAI system using
fully serverless AWS services. It focuses on cost optimization, scalability,
and hallucination-safe Retrieval-Augmented Generation (RAG) design. 

