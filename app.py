import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from PyPDF2 import PdfReader
import pickle
import os

@st.cache_resource
def load_pdfs(pdf_paths):
    """Parse hardcoded PDFs and extract their text."""
    documents = []
    for pdf_file in pdf_paths:
        reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in reader.pages])
        documents.append(text)
    return documents

@st.cache_resource
def initialize_vector_store(pdf_texts):
    """Create embeddings and store them in FAISS for similarity search."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [chunk for doc in pdf_texts for chunk in text_splitter.split_text(doc)]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding_model)

    with open("vector_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

@st.cache_resource
def initialize_qa_chain(_vector_store):
    """Set up the retrieval-augmented QA chain."""
    retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        max_length=1024,
        max_new_tokens=200,
        pad_token_id=50256
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit Interface
st.title("10-K Document QA System")
st.write("Ask questions about the financial documents (Google, Tesla, Uber).")

# Step 1: Hardcoded PDF Paths
pdf_files =  ["/content/drive/MyDrive/ALMENO_AI/goog-10-k-2023 (1).pdf",  # Path to Google 10-K PDF
    "/content/drive/MyDrive/ALMENO_AI/tsla-20231231-gen.pdf",   # Path to Tesla 10-K PDF
    "/content/drive/MyDrive/ALMENO_AI/uber-10-k-2023.pdf" ]    # Path to Uber 10-K PDF


# Step 2: Process PDFs and vector store
if os.path.exists("vector_store.pkl"):
    with open("vector_store.pkl", "rb") as f:
        vector_store = pickle.load(f)
    st.write("Vector store loaded from saved data!")
else:
    pdf_texts = load_pdfs(pdf_files)
    vector_store = initialize_vector_store(pdf_texts)
    st.write("Vector store created and saved successfully!")

# Step 3: Initialize QA Chain
qa_chain = initialize_qa_chain(vector_store)

# Step 4: User Query
query = st.text_input("Enter your question:", " ")
if st.button("Get Answer"):
    with st.spinner("Retrieving the best answer..."):
        response = qa_chain.run(query)
    st.write("### Response:")
    # st.write(response)

    # Extract the Helpful Answer part from the response
    start = response.find("Helpful Answer:")
    if start != -1:
        helpful_answer = response[start + len("Helpful Answer:"):].strip()
        st.write(helpful_answer)
    else:
        print("Helpful Answer not found.")
