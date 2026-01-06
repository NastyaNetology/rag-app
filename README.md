## Local AI Assistant (RAG-Based)

A simple local AI assistant built to demonstrate how **Retrieval-Augmented Generation (RAG)** works in practice, using internal data and a local Large Language Model (LLM).


## Setup & Requirements

This project runs entirely **locally** and does not require cloud services or a GPU.

### Required Tools

- **Python 3.9+**  

- **Qwen2 0.5B Instruct (GGUF format)**  
  https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF  
  Download the GGUF model file and place it in your local models directory.

- **Python Libraries**  
  - FastAPI  
  - Pandas  
  - NumPy  


## High-Level Architecture

- User sends a question  
- Orchestrator analyzes the request  
- RAG retrieves relevant data (top-K)  
- LLM generates a response using the provided context  


## Inspiration & Learning Resource

**Introduction to Retrieval-Augmented Generation**  
by Alfredo Deza  
https://github.com/alfredodeza/learn-retrieval-augmented-generation
