import os
import json
import time
import requests
from faster_whisper import WhisperModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. Voice Transcription Pipeline (CPU Optimized)
# ==========================================
def transcribe_audio(audio_path, output_path):
    print("\n[1/4] Loading Faster-Whisper model...")
    # OPTIMIZED FOR CPU: device="cpu" and compute_type="int8"
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print(f"      Transcribing {audio_path.split('/')[-1]}...")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    transcript_text = ""
    words_data = []
    
    for segment in segments:
        transcript_text += segment.text + " "
        for word in segment.words:
            words_data.append({
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            })
            
    transcript_text = transcript_text.strip()
    
    # EDGE CASE HANDLING: Empty Audio
    if not transcript_text:
        print("\n[WARNING] No speech detected in the audio file.")
        return {"transcript_text": "", "confidence_scores": []}
            
    output_json = {
        "transcript_text": transcript_text,
        "confidence_scores": words_data
    }
    
    # Save the structured JSON output locally
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)
        
    print(f"      Transcription saved to {output_path}")
    return output_json

# ==========================================
# 2. RAG Engine: Ingestion & Retrieval
# ==========================================
def build_rag_engine(pdf_path, db_path):
    print("\n[2/4] Loading and chunking PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Chunking Strategy: 500 chars with 50 overlap, splitting cleanly at Q headers
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\nQ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print("      Initializing embedding model and ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    return vector_store

# ==========================================
# 3. Local LLM Generation via Ollama
# ==========================================
def query_local_llm(context, user_query):
    # API call to the locally running Ollama server
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""You are a helpful Investor Support Assistant for Sunrise Asset Management.
    Use the following FAQ context to answer the user's query. 
    You MUST cite the specific FAQ question number (e.g., Q1, Q4) as your source in your answer.
    If the answer is NOT explicitly contained in the context, you must politely state that you do not know and refuse to answer. Do not guess.
    
    Context:
    {context}
    
    User Query: {user_query}
    
    Answer:"""
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}. Is the server running?"

# ==========================================
# 4. Pipeline Orchestration
# ==========================================
def run_pipeline(audio_path, pdf_path, output_dir, db_path):
    print("==========================================")
    print("   Starting Voice-to-RAG Pipeline (CPU)   ")
    print("==========================================")
    
    # 1. Transcribe Voice
    transcript_path = os.path.join(output_dir, "transcript.json")
    transcript_data = transcribe_audio(audio_path, transcript_path)
    query_text = transcript_data["transcript_text"]
    
    if not query_text:
        return print("\nPipeline Halted: Audio input was empty.")
        
    print(f"\n[Transcribed Query]: '{query_text}'\n")
    
    # 2. Build/Load RAG
    vector_store = build_rag_engine(pdf_path, db_path)
    
    # 3. Retrieve relevant chunks
    print("\n[3/4] Retrieving relevant context...")
    docs = vector_store.similarity_search(query_text, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 4. Generate Answer
    print("\n[4/4] Generating grounded response via Llama 3 (This may take a moment on CPU)...")
    start_time = time.time()
    final_answer = query_local_llm(context, query_text)
    latency = time.time() - start_time
    
    print("\n==========================================")
    print("             FINAL RESPONSE               ")
    print("==========================================")
    print(final_answer)
    print(f"==========================================\n[Inference Latency: {latency:.2f} seconds]\n")
    
    return final_answer

# ==========================================
# 5. Evaluation Loop
# ==========================================
def run_manual_evaluation(db_path):
    print("\n==========================================")
    print("      Initiating Manual RAG Evaluation    ")
    print("==========================================\n")
    
    print("Loading existing ChromaDB Vector Store for evaluation...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    test_cases = [
        {
            "type": "Standard Query",
            "query": "What is the minimum amount I can invest in a SIP?",
            "expected_citation": "Q4"
        },
        {
            "type": "Edge Case (Specific Rule)",
            "query": "What happens if my bank account doesn't have enough money for the SIP twice in a row?",
            "expected_citation": "Q6"
        },
        {
            "type": "Out of Scope",
            "query": "What is the current stock price of Reliance Industries?",
            "expected_citation": None
        }
    ]
    
    total_latency = 0
    
    for i, test in enumerate(test_cases):
        print(f"Test {i+1} | {test['type']}")
        print(f"Query: '{test['query']}'")
        
        start_time = time.time()
        
        docs = vector_store.similarity_search(test["query"], k=2)
        context = "\n".join([doc.page_content for doc in docs])
        answer = query_local_llm(context, test["query"])
        
        latency = time.time() - start_time
        total_latency += latency
        
        print(f"Response: {answer}")
        print(f"Latency: {latency:.2f} seconds")
        
        print("Checks:")
        if test["expected_citation"]:
            if test["expected_citation"] in answer:
                print(f"  ✅ Correct Source Cited ({test['expected_citation']})")
            else:
                print(f"  ❌ Missing Source Citation ({test['expected_citation']})")
        else:
            print("  ✅ Handled Out-of-Scope gracefully (verify response above)")
            
        print("-" * 40 + "\n")
        
    print(f"\n📊 Average LLM Generation Latency (CPU): {(total_latency / len(test_cases)):.2f} seconds")

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Define relative paths based on your Ubuntu project structure
    BASE_DIR = "."
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_PATH = os.path.join(DATA_DIR, "chroma_db")
    
    AUDIO_FILE = os.path.join(INPUT_DIR, "investor_sample.mp3")
    PDF_FILE = os.path.join(INPUT_DIR, "SunriseAMC_FAQ.pdf")
    
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Verify input files exist before running
    if not os.path.exists(AUDIO_FILE) or not os.path.exists(PDF_FILE):
        print("ERROR: Please ensure 'investor_sample.mp3' and 'SunriseAMC_FAQ.pdf' are placed in the 'input/' directory.")
    else:
        # Run the main pipeline
        run_pipeline(
            audio_path=AUDIO_FILE,
            pdf_path=PDF_FILE,
            output_dir=OUTPUT_DIR,
            db_path=DB_PATH
        )
        
        # Run the evaluation tests
        run_manual_evaluation(db_path=DB_PATH)
