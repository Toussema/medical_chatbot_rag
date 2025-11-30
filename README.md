# Medical RAG Chatbot with Groq & Streamlit

A fast, accurate, and private medical information chatbot powered by **Retrieval-Augmented Generation (RAG)**.

- Uses your own medical PDFs (here: *The Gale Encyclopedia of Medicine*)
- Runs 100 % locally (vector database + embeddings)
- Answers instantly via **Groq** (Llama 3.3 70B at >300 tokens/s)
- Beautiful web interface with **Streamlit**
- Never hallucinates: if it doesn’t know, it says “I don’t know”

## Features

- PDF → Text chunks → FAISS vector store
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (light & fast)
- LLM: **Llama-3.3-70B** via Groq (free tier works perfectly)
- Console version + full Streamlit web app
- Sources shown only when relevant (collapsible expander)
- Strict instruction: answer ONLY from the provided documents

## Project Structure
medical_chatbot_rag/
│

├── data/                              # ← Put your medical PDFs here

│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf

│

├── vectorstore/

│   └── db_faiss/                      # ← Created automatically

│       ├── index.faiss

│       └── index.pkl

│

├── create_memory_for_llm.py           # Step 1: Build the vector database (run once or when PDFs change)

├── connect_memory_with_llm.py         # CLI version (quick testing)

├── medibot.py                         # Streamlit web app (main interface)

├── .env                               # Your secrets (never commit!)

├── .gitignore

└── README.md


## Requirements & Installation

```bash
# 1. Clone the repo
git clone https://github.com/Toussema/medical_chatbot_rag.git
cd medical_chatbot_rag

# 2. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# ou
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
# or manually:
pip install streamlit langchain langchain-community langchain-groq \
             langchain-huggingface langchain-classic faiss-cpu \
             pypdf python-dotenv sentence-transformers

## Get your free Groq API key

```bash
# 1. Go to Groq console
# https://console.groq.com/keys

# 2. Create a new key

# 3. Create a .env file at the project root
echo "GROQ_API_KEY=your_key_here_never_share_it" > .env

## How to Run
# 1. Build the vector database (only once or when you add new PDFs)
python create_memory_for_llm.py

# You should see output like:
# Length of PDF pages: 759
# Length of Text Chunks: 7080

# 2. Test in console (optional)
python connect_memory_with_llm.py

# 3. Launch the web app
streamlit run medibot.py

# Open your browser at:
# http://localhost:8501
