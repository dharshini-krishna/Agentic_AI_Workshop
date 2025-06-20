import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_fixed

# Page Configuration
st.set_page_config(page_title="Empathy-Driven Idea Refinement AI", page_icon="🚀", layout="wide")

# 🎨 Theme Selection
theme = st.sidebar.selectbox(
    "Choose Theme Color 🎨",
    ("Sky Blue", "Purple Pink", "Green Lime", "Dark Mode")
)

# Theme Color Setup
if theme == "Sky Blue":
    background = "linear-gradient(to right, #ffffff, #e3f2fd)"
    primary_color = "#1e88e5"
    text_color = "#1a1a1a"  # Dark text
elif theme == "Purple Pink":
    background = "linear-gradient(to right, #ffffff, #fce4ec)"
    primary_color = "#ab47bc"
    text_color = "#1a1a1a"  # Dark text
elif theme == "Green Lime":
    background = "linear-gradient(to right, #ffffff, #e8f5e9)"
    primary_color = "#43a047"
    text_color = "#1a1a1a"  # Dark text
elif theme == "Dark Mode":
    background = "linear-gradient(to right, #1e1e1e, #2c2c2c)"
    primary_color = "#00e676"
    text_color = "#f5f5f5"  # Light text
else:
    background = "linear-gradient(to right, #ffffff, #e3f2fd)"
    primary_color = "#1e88e5"
    text_color = "#1a1a1a"

# Dynamic CSS
def set_custom_css():
    st.markdown(f"""
        <style>
            .stApp {{
                background: {background};
                color: {text_color};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            h1, h2, h3, h4 {{
                color: {primary_color};
            }}
            div.stButton > button {{
                background-color: {primary_color};
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }}
            div.stButton > button:hover {{
                background-color: #0b79d0;
                transform: scale(1.02);
            }}
            .stSpinner > div > div {{
                border-top-color: {primary_color} !important;
            }}
            textarea, input {{
                border: 2px solid {primary_color} !important;
                border-radius: 8px;
                padding: 8px;
                color: {text_color};
                background-color: #ffffff;
            }}
            hr {{
                border: 1px solid {primary_color};
            }}
            .custom-output {{
                color: {text_color};
                background-color: rgba(255, 255, 255, 0.6);
                padding: 15px;
                border-radius: 8px;
            }}
            .dark-mode-output {{
                color: {text_color};
                background-color: rgba(50, 50, 50, 0.8);
                padding: 15px;
                border-radius: 8px;
            }}
        </style>
    """, unsafe_allow_html=True)

set_custom_css()

# Google Gemini 1.5 Flash Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyDmhmpQrQczwfRdp4afpd201VcLmbjPPEo"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Retry-safe Gemini call
@retry(stop=stop_after_attempt(5), wait=wait_fixed(15))
def safe_llm_call(prompt):
    try:
        return llm([HumanMessage(content=prompt)]).content
    except ResourceExhausted as e:
        st.warning("Rate limit hit. Waiting 10 seconds before retrying...")
        raise e

# Agent 1: Idea Purpose Clarifier
def idea_purpose_clarifier(input_text):
    prompt = f"""You are an idea clarifier. Rewrite the following student idea clearly stating what it does, who it is for, and why it matters: {input_text}"""
    return safe_llm_call(prompt)

# Agent 2: Idea Comparator (Vector Similarity)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_empathy_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

idea_comparator = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Agent 3: Empathy Insight Collector
def empathy_insight_collector(feedback_text):
    prompt = f"""You are an empathy-driven agent. Extract 3 actionable emotions or unmet needs from the following user feedback: {feedback_text}"""
    return safe_llm_call(prompt)

# Agent 4: Idea Ranking Agent (Dynamic)
def idea_ranker(input_text):
    prompt = f"""You are an idea ranking agent. Carefully read the following sections and score the idea based on Empathy Fit, Innovation, Feasibility, and Clarity (1-10 for each). Provide a brief justification for each score and an overall ranking.

{input_text}

Start your response now:"""
    return safe_llm_call(prompt)

# Agent 5: Empathy Journal Generator (Tracks Feedback Sources)
def empathy_journal_generator(input_text):
    prompt = f"""You are an empathy journal generator. Summarize the design evolution process for this idea based on the provided details.

{input_text}

Track specific interaction points like interviews, usability tests, and design iterations. Provide future improvement suggestions based on these feedback sources."""
    return safe_llm_call(prompt)

# LangChain Tools
tools = [
    Tool(name="IdeaPurposeClarifier", func=idea_purpose_clarifier, description="Clarifies the student's project idea."),
    Tool(name="IdeaComparator", func=idea_comparator.run, description="Compares the student's idea with existing market projects."),
    Tool(name="EmpathyInsightCollector", func=empathy_insight_collector, description="Collects emotional insights from user feedback."),
    Tool(name="IdeaRanker", func=idea_ranker, description="Ranks the idea based on empathy and innovation."),
    Tool(name="EmpathyJournalGenerator", func=empathy_journal_generator, description="Generates an empathy journal that tracks idea evolution.")
]

# Initialize Agent
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)

# Initialize Feedback History in Session State
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

if 'feedback_sources' not in st.session_state:
    st.session_state.feedback_sources = []

# Streamlit UI
st.title("🚀 Empathy-Driven Idea Refinement AI")
st.markdown("---")

st.markdown("### 💡 Enter Your Raw Student Idea")
input_text = st.text_area("Describe your idea here:")

st.markdown("### 🗣️ Enter User Feedback")
feedback_text = st.text_area("Provide feedback here:")

st.markdown("### 📌 Feedback Source")
feedback_source = st.text_input("Where did this feedback come from? (Interview, Usability Test, Survey, etc.)")

st.markdown("### 🛠️ Design Iteration Notes (Optional)")
iteration_notes = st.text_area("Enter any design iteration notes here:")

if st.button("🚀 Run Multi-Agent Flow"):
    try:
        if feedback_text:
            st.session_state.feedback_history.append(feedback_text)
        if feedback_source:
            st.session_state.feedback_sources.append(feedback_source)
        else:
            st.session_state.feedback_sources.append("Unknown source")

        combined_feedback = " ".join(st.session_state.feedback_history)
        combined_sources = "; ".join(st.session_state.feedback_sources)

        with st.spinner("✨ Clarifying the idea..."):
            clarified = agent.run({"input": f"Clarify this idea: {input_text}", "chat_history": []})

        with st.spinner("🔍 Comparing the idea with existing solutions..."):
            comparator = agent.run({"input": f"Compare this idea: {input_text}", "chat_history": []})

        if combined_feedback:
            with st.spinner("💬 Extracting empathy insights from all feedback..."):
                empathy = agent.run({"input": f"Extract empathy insights: {combined_feedback}", "chat_history": []})
        else:
            st.warning("⚠️ No user feedback provided. Empathy Insights will be skipped.")
            empathy = "No feedback provided."

        with st.spinner("🏆 Ranking the idea dynamically..."):
            ranking = agent.run({"input": f"Clarified Idea: {clarified}\n\nEmpathy Insights: {empathy}\n\nComparator Summary: {comparator}", "chat_history": []})

        with st.spinner("📔 Generating the empathy journal..."):
            journal = agent.run({"input": f"Clarified Idea: {clarified}\n\nEmpathy Insights: {empathy}\n\nFeedback Sources: {combined_sources}\n\nIteration Notes: {iteration_notes}", "chat_history": []})

        st.success("✅ Multi-Agent Flow Completed!")

        output_class = "custom-output" if theme != "Dark Mode" else "dark-mode-output"

        st.markdown("---")
        st.subheader("✨ Refined Idea")
        st.markdown(f'<div class="{output_class}">{clarified}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔍 What We Found in Similar Ideas")
        st.markdown(f'<div class="{output_class}">{comparator}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💬 What Users Are Feeling (Empathy Insights)")
        st.markdown(f'<div class="{output_class}">{empathy}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🏆 Idea Evaluation Score")
        st.markdown(f'<div class="{output_class}">{ranking}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📔 Design Journey & Next Steps")
        st.markdown(f'<div class="{output_class}">{journal}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🗂️ Where the Feedback Came From")
        st.markdown(f'<div class="{output_class}">{combined_sources}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
