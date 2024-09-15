import sys
import importlib

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import urllib.parse
import time
import chromadb

# Load environment variables
load_dotenv()

# Use environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Initialize Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0827",
    generation_config=generation_config,
)

# Initialize OpenAI Embeddings with the specific model and batch size
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
    chunk_size=1000  # Process 1000 texts at a time
)

# Initialize Chroma persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize Chroma vector store
vectorstore = Chroma(
    client=chroma_client,
    collection_name="cc_search",
    embedding_function=embeddings
)

def rewrite_query(question):
    prompt = f"""
    You are a helpful assistant that rewrites user questions into concise search queries
    Your goal is to help the user search the Colorado College website. It is currently the 2024-25 academic year.
    Rewrite the following question as a short, concise search query suitable for a search engine. 
    The query should be no more than 10 words long and focus on the key information needed.
    Do not include any explanations or multiple options. Just provide the single best search query.

    Question: {question}

    Search Query:"""
    
    response = model.generate_content(prompt)
    rewritten_query = response.text.strip()
    print(f"Original question: {question}")
    print(f"Rewritten query: {rewritten_query}")
    return rewritten_query

def google_search(query, num_results=5):
    encoded_query = urllib.parse.quote(query)
    url = f"https://customsearch.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={encoded_query}&num={num_results}&fileType=-pdf"
    response = requests.get(url)
    print(f"Google Search URL: {url}")
    print(f"Google Search Response Status Code: {response.status_code}")
    if response.status_code == 200:
        results = json.loads(response.text)
        items = [item for item in results.get('items', []) if not item['link'].lower().endswith('.pdf')]
        items = items[:5]  # Limit to top 5 results
        print(f"Number of search results (excluding PDFs): {len(items)}")
        return items
    else:
        print(f"Search request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")
        raise Exception(f"Search request failed with status code: {response.status_code}")

def scrape_and_parse(url):
    print(f"Scraping URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.select_one('div.container.cc-subsite-content')
    if content_div:
        text = content_div.get_text(strip=True)
    else:
        text = soup.get_text(strip=True)
    print(f"Scraped text length: {len(text)} characters")
    return text

def process_search_results(results):
    texts = []
    source_urls = []
    for item in results:
        url = item.get('link')
        if url:
            try:
                content = scrape_and_parse(url)
                texts.append(content)
                source_urls.append(url)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue

    print(f"Number of successfully processed URLs: {len(source_urls)}")
    if texts:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents(texts)
        print(f"Number of documents created: {len(docs)}")
        
        # Add documents in one batch
        with st.spinner("Adding documents to vector store..."):
            vectorstore.add_documents(docs)
        print("All documents processed and added to vector store")
    else:
        print("No texts were successfully processed")

    return source_urls

def generate_answer(question, context):
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    print(f"Prompt for final answer: {prompt}")
    response = model.generate_content(prompt)
    answer = response.text
    print(f"Generated answer length: {len(answer)} characters")
    return answer

def generate_followup_questions(question, answer):
    prompt = f"Based on the question '{question}' and the answer '{answer}', generate 3 relevant follow-up questions:"
    response = model.generate_content(prompt)
    followup_questions = response.text.split('\n')
    print(f"Number of follow-up questions generated: {len(followup_questions)}")
    return followup_questions

def main():
    st.title("Ask CC")
    
    question = st.text_input("Enter your question:")
    
    if st.button("Search"):
        with st.spinner("Searching for an answer..."):
            print(f"Processing question: {question}")
            search_query = rewrite_query(question)
            search_results = google_search(search_query)
            
            if not search_results:
                st.write("No search results found. Please try a different question.")
                print("No search results found.")
                return
            
            progress_text = st.empty()
            progress_text.text("Processing search results...")
            source_urls = process_search_results(search_results)
            
            if not source_urls:
                st.write("No valid search results found.")
                print("No valid search results found.")
                return

            progress_text.text("Retrieving relevant documents...")
            relevant_docs = vectorstore.similarity_search(question, k=3)
            print(f"Number of relevant documents retrieved: {len(relevant_docs)}")
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            progress_text.text("Generating answer...")
            answer = generate_answer(question, context)
            followup_questions = generate_followup_questions(question, answer)
            
            progress_text.empty()
            st.write("Answer:", answer)
            st.write("Sources:")
            for url in source_urls:
                st.write(url)
            
            st.write("Follow-up questions:")
            for q in followup_questions:
                st.write(q)

if __name__ == "__main__":
    main()