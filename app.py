import sys
import importlib
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
import re

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
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
    try:
        response = model.generate_content(prompt)
        rewritten_query = response.text.strip()
        print(f"Original question: {question}")
        print(f"Rewritten query: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"Error in rewrite_query: {e}")
        raise

@sleep_and_retry
@limits(calls=100, period=100)
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

@lru_cache(maxsize=100)
def cached_google_search(query, num_results=5):
    return google_search(query, num_results)

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

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(question, context, sources):
    # Modify the prompt to more strongly emphasize proper citation usage
    prompt = f"""You are a helpful assistant that answers questions about Colorado College based on information from the CC website. It is currently the 2024-25 academic year. 

Question: {question}

Context: {context}

Instructions:
1. If you have enough information to answer the question confidently and accurately, provide a direct answer.
2. When using information from the context, cite ONLY the specific source that contains the information you're referencing.
3. Each citation should point to exactly one source where that specific information came from.
4. Do NOT combine multiple source numbers in a single citation unless that exact piece of information appears in multiple sources.
5. If you don't have enough information to answer the question appropriately, say so.
6. Do NOT make up information or guess if you're unsure.
7. Don't add any HTML tags to your response.

Example of CORRECT citation usage:
- Information from first source [1]
- Different information from second source [2]
- Another fact from first source [1]

Example of INCORRECT citation usage:
- Don't cite multiple sources unless necessary [1, 2, 3]
- Don't cite sources that don't contain the specific information

Now, please answer the given question using the provided context and following these citation instructions carefully.

Answer:"""

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Remove LaTeX delimiters
        answer = re.sub(r'\$([^$]+)\$', r'\1', answer)
        
        # Check for insufficient information
        insufficient_info_phrases = [
            "don't have enough information",
            "don't have sufficient information",
            "provided text does not",
            "cannot answer this question",
            "do not have enough information"
        ]
        insufficient_info = any(phrase in answer.lower() for phrase in insufficient_info_phrases)
        
        used_sources = []
        
        if not insufficient_info:
            # Find all unique sources that were actually used
            source_numbers = set()
            citation_pattern = r'\[(?:\d+(?:\s*,\s*\d+)*)\]'
            citations = re.finditer(citation_pattern, answer)
            
            for citation_match in citations:
                citation = citation_match.group(0)
                numbers = [int(num) for num in re.findall(r'\d+', citation)]
                for num in numbers:
                    if 1 <= num <= len(sources):
                        source_numbers.add(num - 1)  # Convert to 0-based index
            
            # Create list of actually used sources
            used_sources = [sources[i] for i in sorted(source_numbers)]
            
            # Replace all citations with the correct source number
            new_answer = answer
            source_map = {old_idx + 1: new_idx + 1 
                         for new_idx, old_idx in enumerate(sorted(source_numbers))}
            
            # Replace complex citations with single citations
            for match in re.finditer(citation_pattern, answer):
                old_citation = match.group(0)
                numbers = [int(num) for num in re.findall(r'\d+', old_citation)]
                # If we have a multi-source citation but only one actual source,
                # replace it with a single citation
                if len(used_sources) == 1:
                    new_citation = "[1]"
                    new_answer = new_answer.replace(old_citation, new_citation)
                else:
                    # Replace with properly numbered citation
                    new_numbers = [source_map[num] for num in numbers if num in source_map]
                    if new_numbers:
                        new_citation = f"[{new_numbers[0]}]"  # Use only first number
                        new_answer = new_answer.replace(old_citation, new_citation)
            
            answer = new_answer
        
        print(f"Generated answer length: {len(answer)} characters")
        print(f"Generated answer: {answer}")
        print(f"Sources found: {used_sources}")
        
        return answer, insufficient_info, used_sources
        
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        raise
    
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=10))
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
        try:
            with st.spinner("Searching for an answer..."):
                print(f"Processing question: {question}")
                search_query = rewrite_query(question)
                search_results = cached_google_search(search_query)
                
                if not search_results:
                    st.warning("No search results found. Please try a different question.")
                    print("No search results found.")
                    return
                
                progress_text = st.empty()
                progress_text.text("Processing search results...")
                docs = process_search_results(search_results)
                
                if not docs:
                    st.warning("No valid search results found. Please try a different question.")
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
        except Exception as e:
            st.error("We're sorry, but we encountered an issue while processing your request. Please try again later or contact support if the problem persists.")
            print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()