
# Sunrise AMC: Voice-Powered Investor Support Assistant

An end-to-end, fully local Voice-to-RAG (Retrieval-Augmented Generation) pipeline designed to handle mutual fund investor queries. This prototype processes audio questions, retrieves relevant context from a knowledge base, and generates grounded, accurate responses—all running completely offline.

## Key Features

*   **Voice Transcription (ASR):** Uses `faster-whisper` (Small model) optimized for fast, accurate speech-to-text.
*   **Intelligent Retrieval (RAG):** Ingests PDF documents using `langchain`, chunks text intelligently, and stores embeddings locally via `ChromaDB` and `sentence-transformers`.
*   **Grounded Generation (LLM):** Queries a locally hosted `Llama 3` (8B) model via `Ollama` to generate answers that strictly cite the provided FAQ document and resist hallucination.
*   **Edge Case Handling:** Proactively catches empty audio inputs and gracefully refuses out-of-scope queries.

## Project Structure

Ensure your project is structured as follows before execution:
```text
sunrise_amc_project/
├── data/               # Auto-generated ChromaDB vector store
├── input/              # REQUIRED: Place input files here
│   ├── SunriseAMC_FAQ.pdf       
│   └── investor_sample.mp3      
├── output/             # Auto-generated transcription JSON logs
├── src/                
│   └── main.py         # Core execution script
├── DECISIONS.md        # Architecture and design tradeoffs
├── README.md           # Documentation
└── requirements.txt    # Python dependencies
```

## Prerequisites & Setup

This pipeline is designed for Ubuntu/Linux environments and requires no paid cloud APIs.

**1. Install System Dependencies**
You will need `zstd` (for extracting Ollama) and Python's virtual environment package.
```bash
sudo apt update
sudo apt install -y zstd curl python3-venv python3-pip
```

**2. Install and Start Ollama (Local LLM Server)**
Ollama serves the Llama 3 model locally.
```bash
# Install Ollama
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Start the server in the background
ollama serve &

# Pull the Llama 3 weights (approx. 4.7GB, may take a few minutes)
ollama pull llama3
```

**3. Setup Python Virtual Environment**
From the root of your project directory (`ml_project/`):
```bash
# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

## Execution

1. **Verify Inputs:** Ensure `investor_sample.mp3` and `SunriseAMC_FAQ.pdf` are located in the `input/` directory.
2. **Run the Pipeline:** With your virtual environment activated and the Ollama server running, execute the main script:
```bash
python src/main.py
```

### What to Expect:
*   The script will transcribe the audio and save the structured output (with word-level timestamps) to `output/transcript.json`.
*   It will chunk and embed the PDF into a local ChromaDB instance inside the `data/` folder.
*   It will query Llama 3 and print the final, grounded response to the console.
*   Finally, it will run an automated **Evaluation Loop** to test system latency, edge cases, and citation accuracy.

## Hardware & Latency Benchmarks

When testing this pipeline, you will notice a significant difference in inference latency depending on the hardware environment. 

During the Google Colab implementation, the pipeline leverages a dedicated NVIDIA T4 GPU, allowing for extremely low latency with the LLM generating responses in **less than 4 seconds per query**. 

Conversely, when running this locally in the Ubuntu terminal using only a standard system CPU (with `int8` quantization applied to the ASR model), the system relies entirely on system RAM and CPU compute. This results in much higher latency, taking approximately **40 seconds per query**. This tradeoff highlights the necessity of GPU infrastructure for scaling this application in a production environment.
