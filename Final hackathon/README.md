
# 🚀 Empathy-Driven Idea Refinement AI

This project is a **multi-agent system** that helps students and startups refine their project ideas by focusing on **empathy-driven design.**  
It uses LangChain agents, Google Gemini 1.5 Flash, and a Retrieval-Augmented Generation (RAG) system to compare ideas, extract empathy insights, and generate design evolution journals.

---

## 🔧 Tech Stack
- 🧩 **LangChain** (Agents, Vector Search, RAG)
- 🤖 **Google Gemini 1.5 Flash API** (LLM)
- 🗂️ **FAISS** (Vector Store for Knowledge Retrieval)
- 🗣️ **HuggingFace Embeddings** (Text Embeddings)
- 🖥️ **Streamlit** (Web App Interface)

---

## 📂 Folder Structure
```plaintext
├── multi_agent_system.py        # Main Streamlit app
├── faiss_empathy_index/         # FAISS vector store folder
├── empathy_feedback_rag.pdf     # Embedded empathy feedback knowledge base
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ✅ Features
- **Idea Clarifier:** Makes raw student ideas clear and focused.
- **Idea Comparator (RAG):** Compares the idea with prior solutions from a custom PDF knowledge base.
- **Empathy Insight Collector:** Extracts emotions, pain points, and unmet needs from user feedback.
- **Dynamic Idea Ranker:** Scores ideas based on empathy, innovation, feasibility, and clarity, using real-time feedback history.
- **Empathy Journal Generator:** Summarizes the design evolution and tracks feedback sources (interviews, usability tests, etc.).
- **Google Gemini Rate Handling:** Retry logic added for API quota management.

---

## 🚀 How to Run
### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/empathy-driven-idea-ai.git
cd empathy-driven-idea-ai
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
streamlit run multi_agent_system.py
```

---

## 🖊️ Example Input
**Student Idea:**
> A smart trash bin that uses AI-powered object detection to automatically sort waste into recyclable, compostable, and landfill categories.

**User Feedback:**
> "Sometimes I don’t know what goes in which bin. I wish there was a system that could help sort it automatically."

**Feedback Source:**
> Usability Test

---

## 🎯 Example Output Sections
- ✨ Refined Idea  
- 🔍 What We Found in Similar Ideas  
- 💬 What Users Are Feeling (Empathy Insights)  
- 🏆 Idea Evaluation Score  
- 📔 Design Journey & Next Steps  
- 🗂️ Where the Feedback Came From  

---

## 📸 Screenshot
*(You can add the screenshot you shared here)*
```plaintext
Insert Screenshot Here
```

---

## ⚠️ API Quota Notes
If you hit this error:
```plaintext
429 You exceeded your current quota.
```
It means you have **exceeded the Google Gemini API free tier limit.**  
Solutions:
- Wait 60 seconds for the quota to reset.
- Add retry handling (already included in the code).
- Upgrade to a paid Gemini plan for higher API limits.

---

## 💻 Built With
- LangChain
- Google Gemini 1.5 Flash API
- FAISS Vector Store
- HuggingFace Embeddings
- Streamlit

---

## ✨ Credits
Developed by Dharshini for the Agentic AI Hackathon.
