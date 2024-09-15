import sys
import importlib
import time
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # If pysqlite3 is not available, we'll use the system sqlite3



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
    model_name="gemini-1.5-flash-exp-0827",
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
    You are a web search query expert who rewrites user questions into concise search queries
    Your goal is to help the user search the Colorado College website. It is currently the 2024-25 academic year, include this in the queryonly if relevant.
    Rewrite the following question as a short, concise search query suitable for a search engine. 
    The query should be brief and will focus on the key information needed. 
    It needs to be expertly crafted to retrieve the most relevant possible results given the users question.
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
        docs = text_splitter.create_documents(texts, metadatas=[{"source": url} for url in source_urls])
        print(f"Number of documents created: {len(docs)}")
        
        # Add documents in one batch
        with st.spinner("Adding documents to vector store..."):
            vectorstore.add_documents(docs)
        print("All documents processed and added to vector store")
    else:
        print("No texts were successfully processed")

    return docs  # Return the documents instead of just the source URLs

def generate_answer(question, context, sources):
    prompt = f"""You are a helpful assistant that answers questions about Colorado College based on information from the CC website. It is currently the 2024-25 academic year. 

Question: {question}

Context: {context}

Instructions:
1. If you have enough information to answer the question confidently and accurately, provide a direct answer.
2. When using information from the context, cite the source using [1], [2], etc. Use each source only once, in the order they appear in the context. ONLY cite sources that are in the context and are used in your answer.
3. If you don't have enough information to answer the question appropriately, respond with a brief statement indicating that you don't have sufficient information to provide an accurate answer.
4. Do NOT make up information or guess if you're unsure.
5. Don't add any HTML tags to your response.

Here are some examples of how to properly cite sources:

Example 1:
Question: What are the housing options for first-year students at Colorado College?
Answer: First-year students at Colorado College are required to live in one of the "Big 3" traditional halls: Mathias Hall, Loomis Hall, or South Hall [1]. These residence halls provide a supportive community environment for new students [2].

Example 2:
Question: How many blocks are in the academic year at Colorado College?
Answer: Colorado College operates on a unique Block Plan, where the academic year consists of 8 blocks [1]. Each block is 3.5 weeks long, allowing students to focus intensively on one subject at a time [2].

Now, please answer the given question using the provided context and following the instructions above.

Answer:"""

    print(f"Prompt for final answer: {prompt}")
    response = model.generate_content(prompt)
    answer = response.text.strip()
    print(f"Generated answer length: {len(answer)} characters")
    print(f"Generated answer: {answer}")
    
    # Check if the answer indicates insufficient information
    insufficient_info = any(phrase in answer.lower() for phrase in [
        "don't have enough information",
        "don't have sufficient information",
        "provided text does not",
        "cannot answer this question",
        "do not have enough information"
    ])
    
    # Extract used sources in the order they were cited
    used_sources = []
    for i, source in enumerate(sources, start=1):
        if f"[{i}]" in answer:
            used_sources.append(source)
    
    return answer, insufficient_info, used_sources

def generate_followup_questions(question, answer):
    prompt = f"""Based on the question '{question}' and the answer '{answer}', generate 2-3 relevant related questions. 
    The user likely to be a prospective student or parent, so think about what questions might be relevant to them.
Output only the questions, one per line, without numbering or explanations.
Example format:
First related question
Second related question
Third related question"""
    response = model.generate_content(prompt)
    followup_questions = response.text.strip().split('\n')
    print(f"Number of follow-up questions generated: {len(followup_questions)}")
    return followup_questions

def main():
    st.title("Ask Colorado College 🐯")
    
    # Use session state to store the current question and a flag for updates
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'update_question' not in st.session_state:
        st.session_state.update_question = False

    # Function to update the question
    def update_question(new_question):
        st.session_state.current_question = new_question
        st.session_state.update_question = True

    # Text input for the question
    question = st.text_input("Enter your question:", value=st.session_state.current_question, key="question_input")
    
    # Search button
    if st.button('Search') or st.session_state.update_question:
        st.session_state.update_question = False
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
            docs = process_search_results(search_results)
            
            if not docs:
                st.write("No valid search results found.")
                print("No valid search results found.")
                return

            progress_text.text("Retrieving relevant documents...")
            relevant_docs = vectorstore.similarity_search(question, k=3)
            print(f"Number of relevant documents retrieved: {len(relevant_docs)}")
            context = "\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(relevant_docs)])
            
            # Safely extract sources, using a default value if 'source' is not in metadata
            sources = [doc.metadata.get('source', f"Source {i+1}") for i, doc in enumerate(relevant_docs)]
            
            progress_text.text("Generating answer...")
            answer, insufficient_info, used_sources = generate_answer(question, context, sources)
            
            progress_text.empty()
            st.write("Answer:", answer)

            followup_questions = generate_followup_questions(question, answer)
            st.write("Related questions:")
            for i, q in enumerate(followup_questions):
                st.button(q, key=f"followup_{i}", on_click=update_question, args=(q,))

            if not insufficient_info and used_sources:
                st.write("Sources used:")
                for i, url in enumerate(used_sources, start=1):
                    st.write(f"[{i}] {url}")

if __name__ == "__main__":
    main()