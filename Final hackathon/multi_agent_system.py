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
input_text = st.text_area("💡 Enter your raw student idea:")
feedback_text = st.text_area("🗣️ Enter user feedback (optional):")
feedback_source = st.text_input("📌 Where did this feedback come from? (interview, usability test, survey, etc.):")
iteration_notes = st.text_area("🛠️ Enter design iteration notes (optional):")

if st.button("Run Multi-Agent Flow"):
    try:
        # Store feedback history and sources
        if feedback_text:
            st.session_state.feedback_history.append(feedback_text)
            if feedback_source:
                st.session_state.feedback_sources.append(feedback_source)
            else:
                st.session_state.feedback_sources.append("Unknown source")

        combined_feedback = " ".join(st.session_state.feedback_history)
        combined_sources = "; ".join(st.session_state.feedback_sources)

        with st.spinner("Clarifying the idea..."):
            clarified = agent.run({"input": f"Clarify this idea: {input_text}", "chat_history": []})

        with st.spinner("Comparing the idea with existing solutions..."):
            comparator = agent.run({"input": f"Compare this idea: {input_text}", "chat_history": []})

        if combined_feedback:
            with st.spinner("Extracting empathy insights from all feedback..."):
                empathy = agent.run({"input": f"Extract empathy insights: {combined_feedback}", "chat_history": []})
        else:
            st.warning("⚠️ No user feedback provided. Empathy Insights will be skipped.")
            empathy = "No feedback provided."

        with st.spinner("Ranking the idea dynamically..."):
            ranking = agent.run({"input": f"Clarified Idea: {clarified}\n\nEmpathy Insights: {empathy}\n\nComparator Summary: {comparator}", "chat_history": []})

        with st.spinner("Generating the empathy journal..."):
            journal = agent.run({"input": f"Clarified Idea: {clarified}\n\nEmpathy Insights: {empathy}\n\nFeedback Sources: {combined_sources}\n\nIteration Notes: {iteration_notes}", "chat_history": []})

        st.success("✅ Multi-Agent Flow Completed!")

        st.subheader("✨ Refined Idea")
        st.write(clarified)

        st.subheader("🔍 What We Found in Similar Ideas")
        st.write(comparator)

        st.subheader("💬 What Users Are Feeling (Empathy Insights)")
        st.write(empathy)

        st.subheader("🏆 Idea Evaluation Score")
        st.write(ranking)

        st.subheader("📔 Design Journey & Next Steps")
        st.write(journal)

        st.subheader("🗂️ Where the Feedback Came From")
        st.write(combined_sources)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

st.info("""
**Agents Used:**
- Clarifier
- Comparator
- Empathy Collector
- Ranker (Dynamic Ranking)
- Empathy Journal Generator (Feedback Tracking)
""")
