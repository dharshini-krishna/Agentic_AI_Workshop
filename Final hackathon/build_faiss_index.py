import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("empathy_feedback_rag.pdf")  # PDF file should be in the same directory
documents = loader.load()

# Step 2: Split the PDF into smaller chunks for embedding
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Embed the chunks using Hugging Face model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)

# Step 4: Save the FAISS index
vectorstore.save_local("faiss_empathy_index")

print("✅ PDF embedded and FAISS index saved successfully!")
