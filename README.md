# Voice-Powered Investor Support Assistant
**Sunrise Asset Management Co. Ltd. — Technical Assessment**

A fully local, CPU-optimised pipeline that accepts a voice query from an investor, transcribes it, retrieves the relevant answer from a structured FAQ knowledge base, and returns a grounded, cited response — all without any paid APIs.

---

## Architecture Overview

```
investor_sample.mp3
        │
        ▼
[1] Faster-Whisper (CPU / int8)
        │  transcript.json
        ▼
[2] RAG Engine
    ├── PyPDFLoader  →  SunriseAMC_FAQ.pdf
    ├── RecursiveCharacterTextSplitter (chunk=500, overlap=50)
    ├── HuggingFace all-MiniLM-L6-v2  →  ChromaDB
        │  top-2 relevant chunks
        ▼
[3] Ollama / Llama 3 (local LLM)
        │
        ▼
   Grounded answer with FAQ citation (e.g. Q9, Q10)
```

---

## Project Structure

```
sunrise-investor-assistant/
├── input/
│   ├── investor_sample.mp3      # Audio query input
│   └── SunriseAMC_FAQ.pdf       # Knowledge base
├── output/
│   └── transcript.json          # Auto-generated transcription
├── data/
│   └── chroma_db/               # Auto-generated vector store (gitignored)
├── main.py                      # Full pipeline + evaluation loop
├── requirements.txt
├── README.md
├── DECISIONS.md
└── .gitignore
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 – 3.11 | 3.12 not tested |
| pip | Latest | `pip install --upgrade pip` |
| Ollama | Latest | [https://ollama.com/download](https://ollama.com/download) |
| Llama 3 model | 8B | Pulled via Ollama (see below) |
| ffmpeg | Any recent | Required by Faster-Whisper for audio decoding |

---

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd sunrise-investor-assistant
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `faster-whisper small` model (~244 MB) and the `all-MiniLM-L6-v2` embedding model (~90 MB) automatically. Ensure you have an active internet connection for this step.

### 3. Install ffmpeg

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH.

### 4. Install Ollama and Pull Llama 3

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Llama 3 8B model (~4.7 GB)
ollama pull llama3

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

> **Hardware note:** The full pipeline runs on CPU. Llama 3 inference on CPU typically takes 30–120 seconds per query on a standard developer laptop. See `DECISIONS.md` for details on this tradeoff.

### 5. Place Input Files

```bash
mkdir -p input output data
cp /path/to/investor_sample.mp3  input/
cp /path/to/SunriseAMC_FAQ.pdf   input/
```

---

## Running the Pipeline

### Full Pipeline (Transcription → RAG → Answer)

```bash
python main.py
```

This single command will:
1. Transcribe `input/investor_sample.mp3` → saves `output/transcript.json`
2. Ingest and embed `input/SunriseAMC_FAQ.pdf` → builds `data/chroma_db/`
3. Retrieve the top-2 most relevant FAQ chunks
4. Generate a grounded, cited answer via Llama 3
5. Run 3 evaluation test cases and print a latency benchmark

### Expected Output (Truncated)

```
==========================================
   Starting Voice-to-RAG Pipeline (CPU)
==========================================

[1/4] Loading Faster-Whisper model...
      Transcribing investor_sample.mp3...
      Transcription saved to output/transcript.json

[Transcribed Query]: 'Hi, I recently invested in an equity mutual fund ...'

[2/4] Loading and chunking PDF...
[3/4] Retrieving relevant context...
[4/4] Generating grounded response via Llama 3 ...

==========================================
             FINAL RESPONSE
==========================================
Based on Q9 and Q10 of the Sunrise AMC FAQ ...
==========================================
[Inference Latency: 47.31 seconds]
```

---

## Evaluation

The pipeline automatically runs a 3-case manual evaluation after the main pipeline:

| Test | Type | Expected Citation |
|---|---|---|
| 1 | Standard Query — SIP minimum | Q4 |
| 2 | Edge Case — consecutive SIP failures | Q6 |
| 3 | Out-of-Scope — stock price query | None (graceful refusal) |

Results are printed to stdout with pass/fail indicators and per-query latency.

---

## Output Files

| File | Description |
|---|---|
| `output/transcript.json` | Word-level transcription with timestamps and confidence scores |
| `data/chroma_db/` | Persisted ChromaDB vector store (gitignored) |

---

## .gitignore

Add the following to avoid committing large binary and generated files:

```
data/chroma_db/
output/
input/*.mp3
input/*.pdf
__pycache__/
*.pyc
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Error connecting to Ollama` | Ensure `ollama serve` is running in a separate terminal |
| `No speech detected` | Verify the MP3 is not silent; try a different audio file |
| Slow inference (>2 min) | Expected on CPU — see `DECISIONS.md`; consider Groq fallback |
| ChromaDB version conflicts | Run `pip install chromadb==0.4.24` to pin to a stable version |
| ffmpeg not found | Install ffmpeg and ensure it is on your system PATH |

---

## License & Disclaimer

*This project is a fictional prototype created for assessment purposes only. Mutual fund investments are subject to market risks.*
