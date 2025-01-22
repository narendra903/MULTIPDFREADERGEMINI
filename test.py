import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

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

# Function to create FAISS vector store
def create_faiss_vectorstore(chunks):
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks]).astype('float32')
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

# Prompt template
def create_prompt(context, question):
    return f"""
    You are an intelligent assistant. Use the context below to answer the question:

    Context: {context}

    Question: {question}

    Answer:
    """

# Streamlit app
def main():
    st.set_page_config(page_title="Multi-PDF Q&A Chatbot", layout="wide")
    st.title("🤖 Multi-PDF Q&A Chatbot using Gemini")

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
                st.write("Extracted Text Preview:", combined_text[:1000])
            st.write(f"### Token Info: Input token limit: {1048576}, Output token limit: {8192}")

            # Split text and create FAISS index
            with st.spinner("Processing text into chunks and generating embeddings..."):
                chunks = split_text(combined_text, max_tokens=300, token_overlap=50)
                faiss_index, chunk_map = create_faiss_vectorstore(chunks)
                st.write(f"Number of chunks added to FAISS: {len(chunks)}")

            st.session_state["faiss_index"] = faiss_index
            st.session_state["chunk_map"] = chunk_map

        else:
            st.sidebar.warning("Please upload at least one PDF file.")

    # Handle Q&A input and responses
    if "faiss_index" in st.session_state:
        faiss_index = st.session_state["faiss_index"]
        chunk_map = st.session_state["chunk_map"]

        st.write("### Ask a Question about the PDF Files:")
        question = st.text_input("Ask a question from the PDF files")
        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Finding the answer..."):
                    # Find the most relevant chunks
                    query_embedding = np.array([embedding_model.encode(question)]).astype('float32')
                    distances, indices = faiss_index.search(query_embedding, k=3)
                    relevant_chunks = [chunk_map[i] for i in indices[0]]
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
