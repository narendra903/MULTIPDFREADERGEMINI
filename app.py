import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import os

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Token limits
MAX_TOKENS_GEMINI = 1048576  # Input token limit for Gemini
MAX_OUTPUT_TOKENS = 8192     # Output token limit for Gemini

# Initialize ChromaDB
# ChromaDB Initialization:
# persist_directory: Specifies where to save the database.
# chroma_db_impl: Uses DuckDB with Parquet for lightweight, high-performance storage.

client = chromadb.Client(Settings(
    persist_directory="chromadb_store",
    chroma_db_impl="duck+parquet",
))

# Load embedding model

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Load summarization model

summarizer = pipeline("summarization",model="facebook/bart-large-cnn")

# Use Hugging Face Hub to download and load the embedding model
def load_embedding_model():
    model_path = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="config.json")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load embedding function
embedding_fn = load_embedding_model()

# Create or load ChromaDB collection

collection_name = "pdf_collection"
if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
else:
    collection = client.get_collection(name=collection_name)

st.set_page_config(page_title="Multi-PDF Q&A", layout="wide")
st.title("📚 Multi-PDF Q&A System with Gemini-1.5-flash-8b")

# Sidebar for PDF upload
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF files here", type=["pdf"], accept_multiple_files=True
)


# Prompt template
def get_prompt(context, question, max_context_tokens):
    # Ensure the context fits within the allowed token limits
    truncated_context = truncate_context(context, max_context_tokens)
    return f"""
    You are an intelligent assistant specializing in extracting insights from documents. Below is the context extracted from the uploaded PDFs, followed by a user question. Use the context to generate an accurate and concise answer.

    Context:
    {truncated_context}

    User Question:
    {question}

    Instructions:
    1. Only use the information provided in the context to answer the question.
    2. If the context does not contain enough information, respond with "The information is not available in the provided documents."
    3. Ensure the response is clear and easy to understand.

    Answer:
    """

# Function to truncate context to fit within token limits
def truncate_context(context, max_context_tokens):
    # Split the context into chunks and calculate token usage
    tokenized_context = context.split()
    if len(tokenized_context) > max_context_tokens:
        return " ".join(tokenized_context[:max_context_tokens])
    return context

# Function to chunk text
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Function to summarize text
def summarize_text(text, max_length=150):
    if len(text.split()) > max_length:
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
    return text

# Function to process PDFs and store embeddings
def process_pdfs_and_store(files, chunk_size=500, use_summarization=True):
    documents = []
    for file in files:
        pdf_reader = PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                chunks = chunk_text(text, chunk_size=chunk_size)
                for chunk in chunks:
                    if use_summarization:
                        chunk = summarize_text(chunk)
                    documents.append({
                        "id": f"{file.name}-page-{page_num}",
                        "text": chunk
                    })
    for doc in documents:
        embedding = embedding_model.encode(doc["text"])
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"source": doc["id"]}],
            ids=[doc["id"]],
        )
    return len(documents)

# Main section for Q&A
st.header("Ask Questions About the PDFs")
query = st.text_area("Ask me anything about the uploaded PDFs:")
submit_button = st.button("Submit")

# Process PDFs and generate embeddings
if uploaded_files:
    st.sidebar.write(f"Uploaded {len(uploaded_files)} PDF(s).")
    with st.spinner("Processing PDFs..."):
        num_documents = process_pdfs_and_store(
            uploaded_files, chunk_size=500, use_summarization=True
        )
    st.sidebar.success(f"Processed and indexed {num_documents} chunks.")

# Handle query submission
if submit_button and query:
    if not uploaded_files:
        st.error("Please upload PDF files first!")
    else:
        with st.spinner("Searching for answers..."):
            # Generate query embedding
            query_embedding = embedding_model.encode(query)
            
            # Retrieve relevant chunks from ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            # Combine retrieved documents into a single context
            context = "\n\n".join([doc for doc in results["documents"][0]])
            
            # Calculate available tokens for context (total - query - response tokens)
            max_context_tokens = MAX_TOKENS_GEMINI - len(query.split()) - MAX_OUTPUT_TOKENS
            
            # Use the prompt template with token truncation
            prompt = get_prompt(context, query, max_context_tokens)
            response = genai.generate_text(
                model="models/gemini-1.5-flash-8b",
                prompt=prompt,
                max_tokens=MAX_OUTPUT_TOKENS
            )
            
            # Display results
            st.success("Here is the response from Gemini:")
            st.write(f"**Question:** {query}")
            st.write(f"**Answer:** {response.text}")
            st.write("### Relevant Context:")
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
                st.write(f"**Source {i}:** {metadata['source']}")
                st.write(doc)



