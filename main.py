import os
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ✅ Replace this with your actual key
openai_api_key = "your-openai-api-key-here"

# Free embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sidebar: Upload / URL input
with st.sidebar:
    st.header("📄 Document Loader")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    url = st.text_input("Or enter a webpage URL")

# App title
st.title("🧠 RAG Bot")
st.markdown("Ask questions about your uploaded PDF or a webpage!")

# Question input (form to submit on Enter)
with st.form("question_form"):
    question = st.text_input("Ask your question here:", placeholder="e.g. What is the main idea?")
    submit_clicked = st.form_submit_button("Ask")

# State
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process document or URL
if uploaded_file or url:
    with st.spinner("Processing document..."):
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
        else:
            loader = WebBaseLoader(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)

        # Vector store and QA chain
        vectorstore = FAISS.from_documents(text_chunks, embedding)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Vectors are ready! You can now ask questions.")

# If submit clicked and chain exists
if submit_clicked and question:
    if st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please upload a PDF or provide a URL first.")
