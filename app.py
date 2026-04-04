import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Chatbot", page_icon="🤖")
st.title("🤖 AI Resume Chatbot (RAG)")
st.write("Upload any PDF and chat with it")

   # -------------------------------
    # LLM
    # -------------------------------
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

if USE_GROQ:
    from langchain_groq import ChatGroq

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found")
        st.stop()

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    st.success("🌐 Running on Groq (Cloud)")

else:
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="mistral", temperature=0)
    st.info("💻 Running on Local Mistral (Ollama)")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


    st.success("✅ PDF processed successfully")


    # -------------------------------
    # Prompt
    # -------------------------------
    prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent HR assistant.
    
    Answer ONLY using the resume content provided below.
    Do not make up information.
    
    If answer is not found, say:
    "Information not available in the resume."
    
    Be clear and concise.
    
    RESUME DATA:
    {context}
    
    QUESTION:
    {query}
    """
    )


    # -------------------------------
    # RAG Chain (NO RetrievalQA)
    # -------------------------------
    rag_chain = (
        {
            "context": retriever,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------------------------------
    # Chat
    # -------------------------------
    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
        st.markdown("### ✅ Answer")
        st.write(response)













