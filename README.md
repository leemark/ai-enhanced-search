# AI-Enhanced Colorado College Search

An intelligent search application that uses AI to provide accurate, context-aware answers to questions about Colorado College. The application combines Google Custom Search, OpenAI embeddings, and Google's Gemini AI to deliver precise responses with source citations.

## Features

- 🔍 Smart query rewriting for optimal search results
- 🤖 AI-powered answer generation with source citations
- 💡 Dynamic follow-up question suggestions
- 📚 Vector-based semantic search using ChromaDB
- 🔗 Automatic web scraping and content processing
- 🎯 Rate limiting and retry mechanisms for API stability

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - Google Gemini AI
  - Google Custom Search Engine
  - OpenAI
- A Google Custom Search Engine ID

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-enhanced-search.git
cd ai-enhanced-search
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your question about Colorado College in the search box

4. View the AI-generated answer, complete with source citations and related follow-up questions

## Technical Details

- Uses Streamlit for the web interface
- Implements ChromaDB for vector storage and similarity search
- Utilizes OpenAI's text-embedding-3-small model for document embeddings
- Employs Google's Gemini 2.0 Flash model for query rewriting and answer generation
- Features intelligent rate limiting and retry mechanisms for API stability
- Implements caching for improved performance

## Project Structure

```
ai-enhanced-search/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .gitignore         # Git ignore file
└── chroma_db/         # Vector database storage
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Colorado College for providing the source content
- OpenAI for embeddings technology
- Google for Gemini AI and Custom Search capabilities
- The Streamlit team for their excellent web framework 