# 1. Environment Setup

import os
import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai  as genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
genai.configure(api_key="GEMINI_API_KEY")
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")

# ........... 2. Text Extraction from PDFs......................................

def extract_text_from_pdfs(uploaded_files):
    combined_text = ""
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                combined_text += page.extract_text() or ""
    return combined_text

#................... 3. Text Chunking ...........................................

'''
    Purpose:
Splits the extracted text into smaller chunks, considering token limits and overlapping words for context continuity.

'''














