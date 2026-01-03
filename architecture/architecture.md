# Architecture – GenAI Document Assistant (Serverless RAG)

This project implements a **fully serverless, event-driven GenAI Document Assistant**
using AWS managed services. The system ingests PDFs, extracts text, generates summaries,
creates vector embeddings, and enables Retrieval-Augmented Generation (RAG) based Q&A
through a simple web UI.

## High-Level Architecture

The system is divided into **two independent flows**:

1. **Document Processing Pipeline (Automatic)**
2. **Question Answering Pipeline (User-driven)**

This separation ensures scalability, cost efficiency, and clean responsibility boundaries.


## 1️⃣ Document Processing Pipeline (Automatic)

### Step-by-step flow:

1. **Document Upload**
   - User uploads a PDF to Amazon S3  
   - Path: `input/pdfs/`
   - This upload is the **initial trigger**

2. **Start Textract Job**
   - `lambda_start_pdf_job`
   - Starts **asynchronous Textract OCR**
   - Stores job metadata in DynamoDB

3. **Amazon Textract (Async OCR)**
   - Extracts text from PDF pages
   - Runs asynchronously for large documents

4. **Textract Poller**
   - `lambda_poll_textract`
   - Periodically checks job status
   - Collects extracted text when completed
   - Writes clean text to:
     ```
     output/text/<doc_id>_text.txt
     ```

5. **Dispatcher (Orchestrator)**
   - `lambda_text_dispatcher`
   - Triggered only when `_text.txt` is created
   - Ensures correct execution order:
     - Summarization (async)
     - Chunking (sync)
     - Embedding (async)

6. **Summarization (GenAI)**
   - `lambda_summarize_bedrock`
   - Uses Amazon Bedrock (Claude)
   - Generates:
     - Human-readable summary
     - Metadata JSON
   - Stored in:
     ```
     output/summaries/
     output/metadata/
     ```

7. **Chunking**
   - `doc-chunker`
   - Splits text into overlapping semantic chunks
   - Stored in:
     ```
     output/chunks/<doc_id>/
     ```

8. **Embedding Generation**
   - `doc-embedder`
   - Uses Amazon Titan Embeddings
   - Converts chunks into vectors
   - Stored as JSON in:
     ```
     output/vectors/<doc_id>_vectors_<timestamp>.json
     ```

## 2️⃣ Question Answering Pipeline (User-driven)

### Flow:

1. **UI (HTML + JavaScript)**
   - Simple client-side interface
   - User enters:
     - `doc_id`
     - Question
     - Answer mode (short / medium / long)

2. **API Gateway**
   - Exposes `/qa` endpoint
   - Receives user queries from UI

3. **QA Lambda (RAG)**
   - `qa_lambda`
   - Embeds user question
   - Loads vectors from S3
   - Performs cosine similarity search
   - Selects top-K relevant chunks
   - Builds a strict RAG prompt
   - Calls Amazon Bedrock (Claude)
   - Returns:
     - Answer
     - Sources (chunk references)

## Design Decisions

- **No OpenSearch**: Vectors stored in S3 to minimize cost
- **Async Textract**: Handles large PDFs reliably
- **Dispatcher Lambda**: Prevents duplicate executions
- **Strict RAG**: Model answers only from retrieved chunks
- **UI decoupled from backend**: Clean client–server separation

## Architecture Diagram

Refer to the architecture.png diagram in the `architecture/` folder:
