import os
import tempfile
import streamlit as st
import pickle
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

openai_api_key = "your-openai-api-key-here"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# File path for saving FAISS index using pickle
PICKLE_PATH = "faiss_vectorstore.pkl"

# Sidebar
with st.sidebar:
    st.header("📄 Document Loader")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    url = st.text_input("Or enter a webpage URL")

st.title("🧠 RAG Bot")
st.markdown("Ask questions about your uploaded PDF or a webpage!")

# Question input
with st.form("question_form"):
    question = st.text_input("Ask your question here:", placeholder="e.g. What is the main idea?")
    submit_clicked = st.form_submit_button("Ask")

# Initialize state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Try loading from pkl file
if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    st.success("✅ Loaded FAISS vectorstore from pickle!")

# Process new file or URL
if uploaded_file or url:
    with st.spinner("Processing document..."):
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
        else:
            loader = WebBaseLoader(url)
        data = loader.load()

        # Custom text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        st.info("🔪 Text Splitting...")
        docs = text_splitter.split_documents(data)

        # Embedding and FAISS index creation
        st.info("📦 Building vector store...")
        vectorstore = FAISS.from_documents(docs, embedding)
        time.sleep(2)

        # Save to pickle
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("✅ Vector store saved as pickle!")

        # Build QA chain
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Answering questions
if submit_clicked and question:
    if st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please upload a PDF or provide a URL first.")
