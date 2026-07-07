# ⚖️ AI Legal Document Analyzer

> A production-oriented AI-powered legal document analysis platform that combines Hybrid RAG, intelligent document processing, LLM-as-a-Judge evaluation, privacy-preserving AI, and modern backend engineering to deliver reliable, evidence-backed legal insights.

---

<p align="center">
  <img src="images/Screenshot .png" width="100%">
</p>

---

## 📌 Overview

Legal documents are often lengthy, unstructured, and filled with complex references, OCR artifacts, and hierarchical sections that traditional Retrieval-Augmented Generation (RAG) pipelines struggle to process effectively.

This project addresses these challenges by building an end-to-end Legal AI platform capable of:

* 📄 Understanding complex legal documents
* 🔍 Retrieving trustworthy legal evidence
* 💬 Maintaining conversational context
* 🔒 Protecting sensitive information
* ⚖️ Reducing hallucinations through retrieval validation
* 📚 Managing multiple legal knowledge sources
* 🌐 Falling back to external legal APIs when local knowledge is insufficient

Instead of simply building a chatbot over PDFs, the focus was on designing a scalable and production-oriented AI system.

---

# ✨ Key Features

### 📄 Intelligent Legal Document Processing

* Docling-based PDF parsing
* Automatic synopsis removal
* OCR artifact cleanup
* Legal citation normalization
* Dynamic Regex generation from Table of Contents
* State Machine-based heading reconstruction
* Metadata enrichment
* Hybrid Chunking

---

### 🔍 Hybrid Retrieval Pipeline

The retrieval pipeline combines multiple retrieval strategies to maximize accuracy.

* Dense Vector Retrieval
* BM25 Sparse Retrieval
* Cross-Encoder Reranking
* Metadata-aware Filtering
* Intelligent Query Routing

---

### ⚖️ LLM-as-a-Judge

Before every response, retrieved chunks are evaluated for:

* Relevance
* Legal context
* Supporting evidence
* Chunk quality
* Confidence score

Decision workflow:

```
PASS
   │
Generate Response

RETRY
   │
Retrieve Better Evidence

FALLBACK
   │
Query External Legal API
```

This significantly improves retrieval reliability while reducing hallucinations.

---

### 🔒 Privacy-First AI

Legal documents often contain sensitive personal information.

To protect user privacy, the system includes:

* GLiNER for PII detection
* Microsoft Presidio for anonymization
* Secure placeholder restoration
* Privacy-preserving LLM inference

---

### 🛡️ AI Safety

The application includes:

* Prompt Injection Guardrails
* Jailbreak Protection
* Redis-based Rate Limiting
* Per-user Token Usage Control

---

### 💬 Conversation Intelligence

* Persistent Chat Sessions
* Multi-turn Conversation Memory
* AI-generated Conversation Titles
* Chat History
* Retrieval Audit Logs
* Confidence Tracking

---

### 👤 User Management

* JWT Authentication
* Role-Based Access Control
* User-specific Knowledge Bases
* User Document Isolation

---

# 🏗️ System Architecture

The platform consists of several independent components:

```
User
   │
React Frontend
   │
FastAPI Backend
   │
Authentication
   │
Document Upload
   │
Document Processing Pipeline
   │
Docling
   │
Cleaning
   │
Hierarchy Reconstruction
   │
Hybrid Chunking
   │
Embeddings
   │
ChromaDB
   │
Hybrid Retrieval
   │
LLM-as-a-Judge
   │
PASS / RETRY / FALLBACK
   │
Groq Llama 3.3
   │
Response
```

---

# 🛠️ Technology Stack

## AI & LLM

* Llama 3.3
* Groq
* LangChain
* Hugging Face

## Document Processing

* Docling
* Regex Engine
* State Machine
* OCR Cleaning

## Retrieval

* ChromaDB
* Dense Retrieval
* BM25
* Cross-Encoder Reranking
* Hybrid RAG

## Privacy

* GLiNER
* Microsoft Presidio

## Backend

* FastAPI
* PostgreSQL
* SQLAlchemy
* Alembic
* JWT Authentication
* Redis

## Frontend

* React

---

# 🚀 Core Workflow

```
Upload Legal Document
          │
          ▼
Document Processing
          │
          ▼
Cleaning & Structure Recovery
          │
          ▼
Hybrid Chunking
          │
          ▼
Embeddings
          │
          ▼
Hybrid Retrieval
          │
          ▼
Cross-Encoder Reranking
          │
          ▼
LLM-as-a-Judge
      │
 PASS / RETRY / FALLBACK
      │
      ▼
Generate Response
```

---

# 🚀 Future Enhancements

* GraphRAG
* Multi-Agent Architecture
* Model Context Protocol (MCP)
* Redis Caching
* Cloud Deployment
* Production Monitoring
* Observability
* Advanced Retrieval Optimization

---

# 💡 What I Learned

Building reliable AI applications goes far beyond integrating an LLM API.

This project strengthened my understanding of:

* AI System Design
* Retrieval Engineering
* Backend Engineering
* Privacy-Preserving AI
* Production-oriented RAG
* AI Safety
* Database Design
* Scalable API Development

---

# 👨‍💻 Author

**Vishal Jaiswal**

AI Engineer | Generative AI | LLMs | RAG | FastAPI | Python

Portfolio: [https://vishalvj-portfolio.lovable.app](https://vishalvj-portfolio.lovable.app)

LinkedIn: [https://www.linkedin.com/in/vishaljaiswalvj/](https://www.linkedin.com/in/vishaljaiswalvj/)

---
