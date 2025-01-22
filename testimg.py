import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tempfile
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

# Function to extract text and images from PDFs


def extract_text_and_images_from_pdfs(uploaded_files):
    combined_text = ""
    images = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Extract text and images using PyMuPDF
        pdf_document = fitz.open(temp_file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            combined_text += page.get_text()

            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_path = f"{tempfile.gettempdir()}/page_{page_num + 1}_img_{img_index + 1}.png"
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                images.append(image_path)

        pdf_document.close()
        os.remove(temp_file_path)

    return combined_text, images
    for i, image in enumerate(images_from_pdf):
            
            image_path = f"{path}/page_{i + 1}.png"
            image.save(image_path, 'PNG')
            images.append(image_path)

    return combined_text, images

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
                combined_text, images = extract_text_and_images_from_pdfs(uploaded_files)
                st.write("Extracted Text Preview:", combined_text[:1000])

                # Display extracted images
                if images:
                    st.write("### Extracted Images:")
                    for img_path in images:
                        st.image(img_path, caption="Extracted Image", use_column_width=True)
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
                    distances, indices = faiss_index.search(query_embedding, k=5)
                    relevant_chunks = [chunk_map[i] for i in indices[0]]
                    relevant_chunks = sorted(relevant_chunks, key=lambda x: '09/06/2022' in x, reverse=True)  # Prioritize chunks with the date
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
