# Empathy-Driven Idea Refinement AI 💡🤖

### Hackathon Project | Built with LangChain, Google Gemini 1.5 Flash, FAISS, and Streamlit

---

## 🚀 Project Overview

This project is a **multi-agent AI system** designed to help students and early-stage startups refine their project ideas by:

- Clarifying the idea purpose
- Comparing with existing solutions using Retrieval-Augmented Generation (RAG)
- Extracting empathy insights from feedback
- Ranking the idea based on key criteria
- Tracking the design journey and feedback sources

It uses Google Gemini 1.5 Flash API, LangChain’s agent orchestration, and FAISS vector search.

---

## 🌟 Problem Statement

> Many student project ideas lack clarity, empathy, or differentiation. This system refines raw ideas using multi-agent collaboration to improve:

- Clarity
- Competitor Awareness
- Emotional Connection
- Ranking and Prioritization
- Design Evolution Tracking

---

## 🛠️ Technology Stack

| Technology | Purpose                |
| ---------- | ---------------------- |
| LangChain  | Multi-agent management |
| Gemini API | LLM Processing         |
| FAISS      | Vector Search (RAG)    |
| Streamlit  | Web UI                 |
| Python     | Backend logic          |

---

## 📂 Project Structure

```text
📁 project-folder/
🔺🔺 multi_agent_system.py          # Main app with agent system and Streamlit UI
🔺🔺 faiss_empathy_index/           # FAISS vector index of the empathy PDF
🔺🔺 empathy_feedback_rag.pdf       # Embedded empathy document
🔺🔺 requirements.txt               # Python dependencies
🔺🔺 README.md                      # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run multi_agent_system.py
```

---

## 🥉 System Workflow

### 🔗 **Flowchart: Multi-Agent Connectivity**

```
User Input (Raw Idea, Feedback, Source)
            │
            ▼
    LangChain Agent System
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 1️⃣ Idea Clarifier       │ → Clarifies the raw idea
 └────────────────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 2️⃣ Idea Comparator      │ → Uses RAG (FAISS + PDF) to compare the idea
 └────────────────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 3️⃣ Empathy Collector    │ → Extracts emotional needs from feedback
 └────────────────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 4️⃣ Idea Ranking Agent   │ → Scores idea based on empathy, innovation, feasibility, clarity
 └────────────────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 5️⃣ Empathy Journal Agent │ → Tracks design evolution and feedback sources
 └────────────────────────────────────────┘
            │
            ▼
Final Refined Output Displayed on Streamlit
```

---

## 👌 Key Features

- ✅ Multi-agent flow using LangChain
- ✅ Google Gemini 1.5 Flash LLM integration
- ✅ FAISS Vector Store for RAG
- ✅ Empathy-focused idea refinement
- ✅ Tracks feedback sources and design iteration notes

---

## ✅ Input Guide

When running the system, you will need to provide:

- **Raw Idea:** Student's initial project idea
- **User Feedback:** Optional feedback text about the idea
- **Feedback Source:** Where the feedback came from (interview, survey, usability test, etc.)
- **Iteration Notes:** Optional notes about changes made to the idea

---

## 💡 Future Improvements

- Dynamic re-ranking when new feedback is added
- Multi-agent orchestration using LangGraph or CrewAI
- Sentiment analysis of feedback
- Voice input support for accessibility

---

## 👤 Project Owner

- Dharshini

---

