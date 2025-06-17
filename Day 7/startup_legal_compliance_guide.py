from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

os.environ['GOOGLE_API_KEY'] = "AIzaSyDmhmpQrQczwfRdp4afpd201VcLmbjPPEo"
loader = PyPDFLoader("Startup_Legal_Compliance_Guide.pdf")
documents = loader.load_and_split()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever()
