import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA

from startup_legal_compliance_guide import retriever
import os

# Google Gemini API Setup
os.environ['GOOGLE_API_KEY'] = "AIzaSyDmhmpQrQczwfRdp4afpd201VcLmbjPPEo"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# ---------------------- Agent 1: Business Model Parsing Agent ----------------------
def parse_business_model(user_input):
    prompt = f"Extract the startup sector and geography from this input: {user_input}"
    response = llm([HumanMessage(content=prompt)])
    return response.content

business_model_tools = [
    Tool(
        name="parse_business_model_tool",
        func=lambda x: parse_business_model(x),
        description="Extracts startup sector and geography."
    )
]

business_model_agent = initialize_agent(
    tools=business_model_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------- Agent 2: Compliance Risk Detection Agent ----------------------
def detect_compliance_risks(user_input):
    prompt = f"List all possible legal and compliance risks for this startup: {user_input}. Include data privacy, IP, licensing, and sector-specific regulations."
    response = llm([HumanMessage(content=prompt)])
    return response.content

compliance_risk_tools = [
    Tool(
        name="compliance_risk_detector_tool",
        func=lambda x: detect_compliance_risks(x),
        description="Identifies legal and compliance risks."
    )
]

compliance_risk_agent = initialize_agent(
    tools=compliance_risk_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------- Agent 3: Jurisdictional Resource Finder Agent (RAG) ----------------------
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def rag_query(user_input):
    return qa_chain.run(f"Provide jurisdiction-specific legal guidelines for this startup: {user_input}")

jurisdictional_resource_tools = [
    Tool(
        name="jurisdictional_resource_finder_tool",
        func=lambda x: rag_query(x),
        description="Finds legal resources using the knowledge base."
    )
]

jurisdictional_resource_agent = initialize_agent(
    tools=jurisdictional_resource_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------- Agent 4: Checklist Generator Agent ----------------------
def generate_checklist(user_input):
    prompt = f"Generate a step-by-step legal compliance checklist for this startup: {user_input}"
    response = llm([HumanMessage(content=prompt)])
    return response.content

checklist_generation_tools = [
    Tool(
        name="checklist_generation_tool",
        func=lambda x: generate_checklist(x),
        description="Generates a legal compliance checklist."
    )
]

checklist_generation_agent = initialize_agent(
    tools=checklist_generation_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------- Streamlit UI ----------------------
st.title("🤖 Multi-Agent Legal & Compliance Risk Identifier for Startups")

user_input = st.text_area("Describe your startup with details about your business model, product, and location:")

if st.button("Run Multi-Agent Analysis"):
    if not user_input:
        st.warning("Please provide a startup description.")
    else:
        with st.spinner("Running multi-agent system..."):
            model_info = business_model_agent.run(user_input)
            risks = compliance_risk_agent.run(user_input)
            resources = jurisdictional_resource_agent.run(user_input)
            checklist = checklist_generation_agent.run(user_input)

        st.subheader("🔍 Business Model & Geography:")
        st.write(model_info)

        st.subheader("🛑 Compliance Risks Identified:")
        st.write(risks)

        st.subheader("📚 Jurisdiction-Specific Legal Resources (from your PDF):")
        st.write(resources)

        st.subheader("✅ Compliance Checklist:")
        st.write(checklist)
