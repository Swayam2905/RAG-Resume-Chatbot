# 🧠 AI Resume Chatbot (RAG-based)

An **AI-powered Resume Question–Answering Chatbot** built using **Retrieval-Augmented Generation (RAG)**.

This application allows users or recruiters to upload resume PDFs and ask natural language questions to get **accurate, context-aware answers** in real time.

---

## 🚀 Features

* 📄 PDF resume ingestion
* ✂️ Recursive text chunking with overlap
* 🔍 Semantic search using FAISS
* 🧠 Context-aware answers using local LLM
* ⚡ Fast responses (~1–2 seconds)
* 🔒 Fully local inference (no API required)
* 💻 Runs on CPU (8GB RAM sufficient)
* 🌐 Interactive Streamlit UI

---

## 🏗️ System Architecture (RAG Pipeline)

1. **Document Loader** – Loads resume PDFs using PyPDFLoader
2. **Text Splitter** – RecursiveCharacterTextSplitter
3. **Embedding Model** – HuggingFace `all-MiniLM-L6-v2`
4. **Vector Store** – FAISS for similarity search
5. **Retriever** – Fetches relevant chunks
6. **LLM** – Mistral via Ollama
7. **Prompt Engine** – ChatPromptTemplate
8. **Output Parser** – StrOutputParser
9. **Frontend** – Streamlit

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** LangChain
* **LLM:** Mistral (via Ollama)
* **Embeddings:** HuggingFace Sentence Transformers
* **Vector DB:** FAISS
* **Frontend:** Streamlit

---

## 💬 Example Queries

* “What skills does the candidate have?”
* “Summarize this resume”
* “What projects are mentioned?”
* “Does the candidate know Python?”

---

## 📸 Snapshots

https://github.com/Swayam2905/RAG-Resume-Chatbot/commit/0e9131bc529772bdb768e362c4a6e07f7cfd176f

---

## 🌐 Live App

https://ai-resume-chatbot-rag-m4wnthlxbmkdrtwqcl4sdz.streamlit.app/

---

## 📦 Installation

```bash
pip install langchain langchain-core langchain-community langchain-huggingface langchain-ollama faiss-cpu sentence-transformers streamlit pypdf
```

---

## ▶️ Run Locally

```bash
# Step 1: Install Ollama
# Download from https://ollama.com

# Step 2: Pull Mistral model
ollama pull mistral

# Step 3: Run the app
python -m streamlit run app.py
```

---

## 📊 Results

* Accurate answers grounded in resume data
* Reduced hallucination using RAG pipeline
* No dependency on external APIs
* Efficient semantic retrieval using FAISS

---

## 🚀 Future Improvements

* Hybrid search (BM25 + vector search)
* Reranking for improved retrieval
* Multi-turn conversational memory
* Deployment with FastAPI + React frontend

---

## 👨‍💻 Author

Swayam Gupta
