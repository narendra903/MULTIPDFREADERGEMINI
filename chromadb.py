import os
import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai  as genai
import chromadb
from chromadb.utils import embedding_functions