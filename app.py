import os
import streamlit as st
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from transformers import pipeline
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Load environment variables
load_dotenv()

DEFAULT_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the summarization model explicitly
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name)

# Function to summarize text
def summarize_text(text):
    # Ensure text is within model input limits
    if len(text) > 512:
        text = text[:512]
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to read PDF content
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Function to summarize and chunk uploaded PDFs
def process_uploaded_pdfs(files):
    summaries = []
    for file in files:
        if file.type == "application/pdf":
            text = read_pdf(file)
            try:
                summary = summarize_text(text)
                summaries.append(f"Summary of {file.name}:\n{summary}")
                file_name = file.name[:-4]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                if os.path.exists(f"{file_name}.pkl"):
                    with open(f"{file_name}.pkl", "rb") as f:
                        st.session_state.vector_store = pickle.load(f)
                else:
                    embeddings = OpenAIEmbeddings()
                    st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{file_name}.pkl", "wb") as f:
                        pickle.dump(st.session_state.vector_store, f)
            except Exception as e:
                st.error(f"Error summarizing {file.name}: {e}")
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            try:
                summary = summarize_text(text)
                summaries.append(f"Summary of {file.name}:\n{summary}")
            except Exception as e:
                st.error(f"Error summarizing {file.name}: {e}")
    return summaries

# Function to summarize a web page
def summarize_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    summary = summarize_text(text)
    return summary

# Function to get OpenAI response
def get_openai_response(prompt, model="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
    openai.api_key = DEFAULT_OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message['content'].strip()

# Function to perform similarity search and get relevant context
def get_relevant_context(query, vector_store, top_k=5):
    docs = vector_store.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# Streamlit UI
st.set_page_config(page_title="DocSum", page_icon="📝")

st.title("DocSum: Document and Web Page Summarizer")
st.sidebar.header("Options")

# Chatbot configuration
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_summaries' not in st.session_state:
    st.session_state.pdf_summaries = []
if 'web_summaries' not in st.session_state:
    st.session_state.web_summaries = []
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = {}

with st.sidebar:
    st.markdown("# Chat Options")

    # Widgets
    model = st.selectbox("What model would you like to use",
                         ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'))
    temperature = st.number_input("Temperature", value=0.7,
                                  min_value=0.0, max_value=1.0, step=0.1)
    max_token_length = st.number_input("Max Token Length", value=1000,
                                       min_value=100, max_value=128000)

    st.markdown('# Upload PDF File')
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')
    if pdf:
        st.session_state.pdf_summaries.extend(process_uploaded_pdfs([pdf]))

web_page_url = st.sidebar.text_input("Enter Web Page URL")

if web_page_url:
    if urlparse(web_page_url).scheme:
        st.header("Web Page Summary")
        summary = summarize_web_page(web_page_url)
        st.write(summary)
        st.session_state.web_summaries.append(f"Summary of web page {web_page_url}:\n{summary}")
    else:
        st.sidebar.error("Please enter a valid URL.")

# Path to the folder containing PDF files (local or OneDrive)
folder_path = st.sidebar.text_input("Enter folder path for PDFs")

if folder_path:
    st.header("PDF Summaries from Folder")
    summaries = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                with open(os.path.join(root, file), 'rb') as f:
                    summaries.extend(process_uploaded_pdfs([f]))
    st.session_state.pdf_summaries.extend(summaries)
    for summary in summaries:
        st.write(summary)

# Display the chat messages stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What questions do you have?"):
    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Generating response ..."):
        # Fetch relevant context from past messages, PDF summaries, and web summaries
        past_messages = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        combined_prompt = f"{past_messages}\nuser: {user_prompt}"

        # Get relevant context from vector store
        context = ""
        if st.session_state.vector_store:
            context += get_relevant_context(user_prompt, st.session_state.vector_store)
        
        context += "\n".join(st.session_state.pdf_summaries)
        context += "\n".join(st.session_state.web_summaries)

        combined_prompt_with_context = f"{context}\n{combined_prompt}"

        llm_response = get_openai_response(combined_prompt_with_context, model=model, max_tokens=max_token_length, temperature=temperature)

        st.session_state.messages.append({"role": "assistant", "content": llm_response})
        with st.chat_message("assistant"):
            st.markdown(llm_response)
