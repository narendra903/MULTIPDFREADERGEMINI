import os
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai


