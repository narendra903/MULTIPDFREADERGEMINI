import os
import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")

# Function to call the Gemini model
def gemini_api_call(prompt, max_tokens=8192, input_token_limit=1048576):
    if len(prompt.split()) > input_token_limit:
        raise ValueError(f"Input token limit exceeded! Reduce your input to below {input_token_limit} tokens.")
    response = gemini_model.generate_content(
        contents=prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": max_tokens}
    )
    if response and hasattr(response, 'text'):
        return response.text
    else:
        raise Exception("Error calling Gemini model: No response or unexpected format.")

# Text embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    combined_text = ""
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                combined_text += page.extract_text() or ""
    return combined_text

# Function to split text into chunks while considering token limits
def split_text(text, max_tokens=8192, token_overlap=500):
    words = text.split()
    token_count = len(words)
    chunks = []

    start = 0
    while start < token_count:
        end = min(start + max_tokens, token_count)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - token_overlap  # Apply overlap

    return chunks

# Function to create ChromaDB collection
def create_chroma_collection(chunks):
    client = chromadb.PersistentClient(path="chroma_storage")
    collection = client.get_or_create_collection(
        name="pdf_chunks",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    )

    # Add chunks to the collection
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            metadatas=[{"chunk_index": i}]
        )
    return collection

# Prompt template
def create_prompt(context, question):
    suggest_format = "If the user asks for information in a tabular format, provide it as a table. If the user prefers explanations, provide detailed paragraphs and bullet points."
    return f"""
    🎉 Hello! I'm your intelligent assistant. Here's what I can do for you:

    - Provide concise, clear, and accurate answers.
    - Offer explanations, bullet points, or tables based on the user's preference.
    - Always deliver information in a professional and structured format.

    Context for this query:
    {context}

    Question Asked by User: {question}

    {suggest_format}

    Answer (in a line gap below):

    """

# Streamlit app
def main():
    st.set_page_config(page_title="Multi-PDF Q&A Chatbot", layout="wide")
    st.title("🤖 Multi-PDF Q&A Chatbot using Gemini and ChromaDB")

    # Sidebar for file upload
    st.sidebar.header("Menu:")
    st.sidebar.write("Upload your PDF Files and Click on the Submit & Process Button")
    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop files here", type=["pdf"], accept_multiple_files=True
    )

    if st.sidebar.button("Submit & Process"):
        if uploaded_files:
            st.sidebar.success("Files uploaded successfully! Processing now...")

            # Extract text from uploaded PDFs
            with st.spinner("Extracting text from PDFs..."):
                combined_text = extract_text_from_pdfs(uploaded_files)
            st.write(f"### Token Info: Input token limit: {1048576}, Output token limit: {8192}")

            # Split text and create ChromaDB collection
            with st.spinner("Processing text into chunks and generating embeddings..."):
                chunks = split_text(combined_text, max_tokens=300, token_overlap=50)
                collection = create_chroma_collection(chunks)
                st.session_state["chroma_collection"] = collection
                st.session_state["chunks"] = chunks
                st.write(f"Number of chunks added to ChromaDB: {len(chunks)}")

        else:
            st.sidebar.warning("Please upload at least one PDF file.")

    # Handle Q&A input and responses
    if "chroma_collection" in st.session_state:
        collection = st.session_state["chroma_collection"]
        chunks = st.session_state["chunks"]

        st.write("### Ask a Question about the PDF Files:")
        question = st.text_input("Ask a question from the PDF files")
        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Finding the answer..."):
                    # Find the most relevant chunks
                    query_result = collection.query(query_texts=[question], n_results=5)
                    relevant_chunks = query_result["documents"]
                    context = " ".join(relevant_chunks)

                    # Create prompt and get answer
                    prompt = create_prompt(context, question)
                    answer = gemini_api_call(prompt, max_tokens=8192)

                    st.markdown(f"<div style='background:#f9f9f9;padding:15px;border-radius:10px;'>{answer}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question!")
    else:
        st.warning("Please process the PDFs first by uploading and clicking 'Submit & Process'.")

if __name__ == "__main__":
    main()
