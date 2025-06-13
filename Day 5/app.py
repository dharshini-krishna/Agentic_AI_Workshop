import streamlit as st
import PyPDF2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI  # Remove or comment this line
from langchain_google_genai import ChatGoogleGenerativeAI # Import the Gemini LLM
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
# --- Initialize Language Model ---
# For Google GenAI (e.g., Gemini):
# You can choose models like "gemini-pro" or "gemini-1.5-flash" depending on your needs.
# "gemini-pro" is generally suitable for text generation.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))

# --- LangChain Prompts ---

SUMMARY_PROMPT_TEMPLATE = """
You are a helpful study assistant. Your task is to summarize the following study material into concise bullet points.
Keep the summary focused on key concepts and important information.

Study Material:
{text}

Summary:
"""

QUIZ_PROMPT_TEMPLATE = """
You are a helpful study assistant. Based on the following summarized study material, generate 2-3 multiple-choice questions.
Each question should have four options (a, b, c, d) and clearly indicate the correct answer.
Ensure the questions cover different aspects of the summarized content.

Summarized Material:
{summary}

Quiz Questions:
"""

# --- LangChain Chains ---

# Summary Chain
summary_prompt = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE, input_variables=["text"])
summary_chain = summary_prompt | llm | StrOutputParser()

# Quiz Generation Chain
quiz_prompt = PromptTemplate(template=QUIZ_PROMPT_TEMPLATE, input_variables=["summary"])
quiz_chain = quiz_prompt | llm | StrOutputParser()

# --- Streamlit Application ---

st.set_page_config(page_title="Study Assistant & Quiz Generator", layout="centered")
st.title("📚 Study Assistant & Quiz Generator")
st.markdown("Upload your PDF study material, and I'll summarize it and generate multiple-choice quiz questions for you!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.info("Processing your document...")
    # Read PDF content
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        study_material = ""
        for page in reader.pages:
            study_material += page.extract_text()

        if not study_material.strip():
            st.warning("Could not extract text from the PDF. Please ensure it's not an image-only PDF.")
        else:
            st.subheader("Extracted Study Material (Partial View):")
            st.code(study_material[:1000] + "...", language="text") # Show a snippet

            st.subheader("📝 Summary:")
            with st.spinner("Generating summary..."):
                summary = summary_chain.invoke({"text": study_material})
                st.write(summary)

            st.subheader("❓ Quiz Questions:")
            with st.spinner("Generating quiz questions..."):
                quiz_questions = quiz_chain.invoke({"summary": summary})
                st.markdown(quiz_questions)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure the uploaded file is a valid PDF and not corrupted.")

st.markdown("---")