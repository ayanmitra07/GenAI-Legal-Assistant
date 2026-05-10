# ⚖️ GenAI Legal Assistant

A Retrieval-Augmented Generation (RAG)-based AI legal assistant built using the Indian Constitution.

This system retrieves relevant constitutional articles using hybrid retrieval techniques and generates context-aware legal responses using a local LLM.

---

# Features

- Hybrid Retrieval (Dense + BM25)
- FAISS Vector Search
- Cross-Encoder Reranking
- Clause-Aware Chunking
- Intent Classification
- Guardrails for Unsafe Queries
- Local LLM Reasoning (Mistral 7B)
- Evaluation Framework
- Streamlit Web Interface

---

# Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- BM25
- Cross Encoder
- Llama.cpp
- Mistral 7B

---

# System Architecture

User Query
↓
Intent Classification
↓
Hybrid Retrieval (FAISS + BM25)
↓
Cross Encoder Reranking
↓
Context Injection
↓
Mistral 7B Reasoning
↓
Final Legal Response

---

# Project Structure

```text
LEGAL_AI_ASSISTANT/
│
├── app/
├── models/
├── output/
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ayanmitra07/GenAI-Legal-Assistant.git
cd LEGAL_AI_ASSISTANT
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# Sample Queries

- What is Article 21?
- Can courts interfere in elections?
- What protections exist against arbitrary arrest?
- Can Parliament amend fundamental rights?

---

# Evaluation Metrics

The project includes evaluation using:

- Recall@K
- Clause Recall@K
- Mean Reciprocal Rank (MRR)

---

# Future Improvements

- FastAPI Backend
- Cloud Deployment
- Chat History
- PDF Upload
- Multi-query Retrieval
- API-based LLM Integration
- Dockerization

---

# Disclaimer

This system is for educational and informational purposes only and does not constitute legal advice.

![alt text](image.png)