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
