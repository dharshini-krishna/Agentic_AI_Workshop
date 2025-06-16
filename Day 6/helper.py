# helper.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ================================
# Streamlit UI Setup
# ================================
st.set_page_config(page_title="📄 Legal & Compliance Risk Identifier with RAG")

st.title("📄 Legal & Compliance Risk Identifier with RAG")
st.write("Ask your legal/compliance questions directly. The system will fetch answers from preloaded rules and Gemini 1.5.")

# ================================
# Preload PDF (RAG Source)
# ================================
# Your preloaded legal PDF (replace with your file path)
pdf_path = "Startup_Legal_Compliance_Guide.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ================================
# Vector Store (FAISS)
# ================================
api_key = "AIzaSyAV6HUVn4IgDMXgB86oOGsBWOjf-Y5J-kk"  # Replace with your Gemini API key

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# ================================
# Gemini 1.5 Setup
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# ================================
# Retrieval QA Chain
# ================================
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ================================
# Search Bar Only
# ================================
user_question = st.text_input("Ask your Legal/Compliance Question:")

if st.button("Get Legal Guidance"):
    if user_question:
        with st.spinner("🔍 Searching in the legal documents..."):
            response = qa_chain.run(user_question)
            st.write("### 📄 AI Agent Response:")
            st.write(response)
    else:
        st.warning("Please enter a question to continue!")
