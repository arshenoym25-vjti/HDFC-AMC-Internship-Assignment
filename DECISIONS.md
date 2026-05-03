# Architecture & Design Decisions

This document outlines the rationale behind the technical choices made for the Sunrise AMC Voice-Powered Support Assistant, including model selection, data chunking, system tradeoffs, and a roadmap for production scaling.

## 1. Model Selection

The assignment mandates a fully local execution environment without reliance on paid cloud APIs. The models were chosen to maximize reasoning and transcription accuracy while respecting the constraints of a single-node setup (specifically tested for CPU compatibility).

*   **Large Language Model (LLM):** `llama3` (8B parameters) via Ollama.
    *   *Why:* Llama 3 8B offers best-in-class reasoning for its size. Using Ollama abstracts away the complexity of manual quantization. By utilizing native quantization, the model easily runs on standard system RAM (or comfortably within a 16GB GPU like a T4) without triggering Out-Of-Memory (OOM) errors.
*   **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`.
    *   *Why:* At only 384 dimensions and ~80MB in size, this model is extremely lightweight and fast. Because the provided FAQ is a dense, direct knowledge base, a massive embedding model (like `bge-large`) would introduce unnecessary latency without a tangible gain in retrieval accuracy.
*   **Automatic Speech Recognition (ASR):** `faster-whisper` (Small).
    *   *Why:* It leverages the `CTranslate2` engine, making it up to 4x faster than OpenAI's standard Whisper implementation. For maximum hardware compatibility, it was configured with `compute_type="int8"`, allowing it to run efficiently on a standard CPU while maintaining high accuracy for conversational audio.

## 2. Chunking Strategy

*   **Method:** `RecursiveCharacterTextSplitter`
*   **Parameters:** `chunk_size=500`, `chunk_overlap=50`
*   **Separators:** `["\nQ", "\n\n", "\n", " ", ""]`
*   **Rationale:** The provided `SunriseAMC_FAQ.pdf` is structured as a direct Q&A. The primary goal of chunking here is to prevent "context bleeding" where an answer to Q4 gets mixed with the premise of Q5. By prioritizing `\nQ` as the primary separator and keeping the chunk size to 500 characters, we ensure that an entire Question and its corresponding Answer remain isolated within a single, cohesive vector space. The 50-character overlap acts as a safety net for edge-case formatting.

## 3. Evaluation & Edge Cases (The "Green Flags")

To ensure robust answer quality and monitor system performance, a manual evaluation loop was integrated directly into the execution script (`main.py`).

*   **Latency Benchmarking:** The script tracks and prints the exact inference time of the LLM generation. (Note: On a CPU, inference takes ~30-45 seconds, whereas on a T4 GPU, it drops to <4 seconds).
*   **Out-of-Scope Handling:** The LLM prompt is strictly constrained. If a user asks about stock prices (not in the FAQ), the prompt forces the model to politely refuse rather than hallucinate.
*   **Empty Audio Protection:** The ASR pipeline explicitly checks for an empty transcription string. If silence or noise is detected with no words, the pipeline halts immediately, preventing the LLM from attempting to answer a null query.

## 4. Tradeoffs Made

*   **Sacrificed Speed for Simplicity & Hardware Constraints:** Running `faster-whisper` and an 8B LLM synchronously on the same machine means that processing is blocked serially. One request must fully finish before the next begins. This was a necessary tradeoff to meet the "run locally on standard hardware" constraint without overcomplicating the setup with local task queues.
*   **In-Memory vs. Persistent Vector Store:** For simplicity, ChromaDB is built and persisted locally in the `./data` folder during execution.

## 5. Production Readiness: What Would NOT Scale

While this architecture serves as a functional prototype, several components would catastrophically fail if deployed at scale (handling thousands of queries an hour):

1.  **Stateful Ingestion:** Currently, the document is loaded and embedded dynamically. At scale, document ingestion (ETL) must be entirely decoupled from the inference pipeline. We would migrate to a managed, remote vector database (e.g., Milvus, Pinecone, or a distributed Qdrant cluster). The inference API should be stateless.
2.  **GPU Bottlenecking (Serial Processing):** Processing audio and generating text sequentially on a single node blocks all other concurrent users. In production, we must utilize an event-driven architecture using a message broker (e.g., Apache Kafka or RabbitMQ) to distribute transcription tasks and LLM tasks to independent, autoscaling worker pools.
3.  **LLM Inference Server:** Ollama is excellent for local development but lacks the throughput optimizations required for production. The LLM backend would be migrated to `vLLM` or `TensorRT-LLM` to utilize continuous batching and PagedAttention, maximizing the tokens generated per second across a GPU cluster.
