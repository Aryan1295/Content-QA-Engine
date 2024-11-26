# Content-QA-Engine
# Streamlit-Based Question Answering System for 10-K Documents

This project implements a **Streamlit-based Question Answering (QA) system** that processes **10-K financial documents** for companies like Google, Tesla, and Uber. The system utilizes **LangChain**, **FAISS**, and **Transformers** to retrieve relevant information and generate answers to user queries.

---

## **Features**

- Upload and process **10-K documents** dynamically or use preloaded PDFs.
- Retrieve relevant context from documents using **FAISS vector search**.
- Generate answers using a local **HuggingFace language model (GPT-2)**.
- Interactive Streamlit-based web interface.

---

## **Technologies Used**

- **Streamlit**: Web application framework for the interface.
- **LangChain**: Framework for combining retriever and language model in a pipeline.
- **FAISS**: Fast similarity search for vectorized document chunks.
- **Transformers**: HuggingFace library for embeddings and text generation.
- **PyPDF2**: Parsing and extracting text from PDF documents.

---

