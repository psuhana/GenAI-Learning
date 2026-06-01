# GenAI-Learning

A hands-on repository documenting my journey of learning and building Generative AI systems from scratch using Python, OpenAI APIs, embeddings, vector databases, semantic retrieval, and Retrieval-Augmented Generation (RAG).
This repository focuses on understanding the internal mechanics of GenAI systems instead of relying entirely on high-level frameworks.

## Repository Structure

### 01_chatbots
- Basic OpenAI chatbot implementations
- Chat completion APIs
- Message history handling

### 02_memory
- Persistent chatbot memory
- JSON-based conversation storage
- Context persistence

### 03_embeddings
- Embedding generation
- Cosine similarity
- Semantic search
- Manual retrieval systems

### 04_rag
- ChromaDB vector database integration
- Persistent vector storage
- Multi-chunk retrieval
- Conversational RAG chatbot
- Metadata tracking
- Hallucination control

## Concepts Explored

- OpenAI Chat Completions
- Embeddings
- Cosine Similarity
- Semantic Retrieval
- Vector Databases
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Persistent Memory Systems
- Context Window Management
- Prompt Engineering
- Conversational AI Systems

## Tech Stack

- Python
- OpenAI API
- ChromaDB
- NumPy
- dotenv

## Setup

1. Clone the repository

2. Install dependencies

pip install -r requirements.txt

3. Create a .env file

OPENAI_API_KEY=your_api_key_here

4. Run any project file

Example:
python 04_rag/conversational_rag_chatbot.py

## Learning Approach

This repository emphasizes building GenAI systems manually from first principles before moving to higher-level orchestration frameworks.

The goal is to deeply understand how:
- embeddings work
- semantic retrieval functions
- vector databases operate
- RAG pipelines are structured
- conversational memory is managed
