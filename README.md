# Session Cloud AI - PDF Question Answering System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over PDF documents using embeddings, vector databases, and AI models.

## 🎯 Overview

This project implements a complete RAG pipeline that:
- Extracts and chunks text from PDF documents
- Generates semantic embeddings for document chunks
- Stores embeddings in a vector database (Chroma)
- Retrieves relevant chunks based on queries
- Uses Google's Gemini API to generate contextual answers

## ✨ Features

- **PDF Processing**: Automatic text extraction and intelligent chunking
- **Semantic Search**: Uses SentenceTransformers for semantic embeddings
- **Vector Database**: Persistent storage with Chroma
- **LangChain Integration**: Simplified RAG chain management
- **Google Gemini Integration**: State-of-the-art LLM for answer generation
- **Configurable Parameters**: Adjustable chunk sizes and overlap for optimization

## 📋 Prerequisites

- Python 3.8+
- PDF files to process
- Google Gemini API key

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Session-Cloud-Ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install pypdf sentence-transformers chromadb langchain langchain-community langchain-google-genai langchain-huggingface
   ```

## 🔧 Configuration

### API Key Setup

Set your Google Gemini API key:

```python
import os
os.environ['GEMINI_API_KEY'] = "your_api_key_here"
```

### Available Implementations

#### Option 1: `main.py` - Basic RAG Pipeline
Direct implementation using:
- PyPDF for text extraction
- SentenceTransformers for embeddings
- Chroma for vector storage

**Usage:**
```bash
python main.py
```

#### Option 2: `session.py` - Advanced LangChain Pipeline
Production-ready implementation using:
- LangChain for orchestration
- Google Generative AI (Gemini)
- HuggingFace embeddings
- Chroma with LangChain integration

**Usage:**
```bash
python session.py
```

## 📖 Usage

### Basic Example (main.py)

```python
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load and extract text from PDF
reader = PdfReader("sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# 2. Chunk the text
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. Generate embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# 4. Store in vector database
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="docs")
collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,
    ids=[f"id_{i}" for i in range(len(chunks))]
)

# 5. Query
query = "What is the main topic?"
query_embedding = model.encode([query])
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
print(results["documents"][0])
```

### Advanced Example (session.py)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=1)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask questions
response = qa_chain.invoke("Summarize the main objectives?")
print(response)
```

## 🏗️ Project Structure

```
Session-Cloud-Ai/
├── main.py              # Basic RAG implementation
├── session.py           # Advanced LangChain implementation
├── chroma_store/        # Persistent vector database
├── chroma_db/           # Alternative vector store location
├── myenv/               # Python virtual environment
├── sample.pdf           # Sample document for processing
└── README.md            # This file
```

## 🔄 How It Works

```
PDF → Text Extraction → Chunking → Embeddings → Vector DB
                                                     ↓
                                        Query Processing ← User Query
                                                     ↓
                                        Similarity Search
                                                     ↓
                                        Context + Query → LLM
                                                     ↓
                                        Generated Answer
```

## 🛠️ Key Components

| Component | Library | Purpose |
|-----------|---------|---------|
| PDF Processing | PyPDF | Extract text from PDF files |
| Text Splitting | LangChain | Intelligent document chunking with overlap |
| Embeddings | SentenceTransformers | Generate semantic vector representations |
| Vector DB | Chroma | Store and retrieve embeddings |
| LLM | Google Gemini | Generate contextual responses |
| Orchestration | LangChain | Manage RAG pipeline flow |

## ⚙️ Customization

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Larger chunks for complex docs
    chunk_overlap=200     # More overlap for better context
)
```

### Change Embedding Model
```python
embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"  # Different model for different use cases
)
```

### Modify LLM Parameters
```python
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,      # Lower for deterministic responses
    top_p=0.9
)
```

## 📊 Performance Tips

1. **Batch Processing**: Process multiple PDFs in parallel
2. **Chunk Optimization**: Balance between chunk size and retrieval precision
3. **Embedding Caching**: Cache embeddings for unchanged documents
4. **Vector DB Indexing**: Chroma automatically creates efficient indexes

## 🤝 Contributing

Feel free to open issues and pull requests for improvements.

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### API Key Issues
- Ensure `GEMINI_API_KEY` environment variable is set correctly
- Check API key permissions in Google Cloud Console

### Memory Issues
- Reduce `chunk_size` for large PDFs
- Process documents in batches

### Embedding Errors
- Verify internet connection for downloading SentenceTransformers models
- Check available disk space for model caching

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)

## ✅ Tested With

- Python 3.12
- ChromaDB 1.5.8
- LangChain Community
- Google Generative AI
- SentenceTransformers

---

**Happy Querying! 🚀**
