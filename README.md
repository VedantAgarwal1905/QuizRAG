# QuizRAG 📚

**An AI-powered quiz generation and self-assessment system built using Retrieval-Augmented Generation (RAG).**

QuizRAG transforms study materials into an interactive learning experience. It uses document processing, semantic search, vector embeddings, and a Large Language Model (LLM) to help students revise concepts and assess their understanding of academic notes.

## 🚀 Overview

Preparing for exams often involves navigating lengthy PDFs and manually creating practice questions. QuizRAG aims to simplify this process by using RAG to retrieve relevant information from study material and generate context-aware responses.

The project demonstrates how document ingestion, text chunking, vector databases, and LLMs can work together in an AI-powered learning application.

## ✨ Key Features

- **Document-based learning:** Process academic study material in PDF format.
- **Text chunking:** Split documents into smaller, manageable chunks for retrieval.
- **Semantic embeddings:** Generate vector representations of text using a Hugging Face embedding model.
- **Vector storage and retrieval:** Store and search document embeddings using ChromaDB.
- **Retrieval-Augmented Generation:** Provide relevant document context to an LLM when generating responses.
- **AI-assisted revision:** Build a foundation for generating practice questions and evaluating preparation based on course material.

## 🧠 How It Works

1. **Document ingestion:** Load academic notes and extract their text.
2. **Text splitting:** Divide the extracted content into overlapping chunks using a recursive text splitter.
3. **Embedding generation:** Convert text chunks into 384-dimensional vector representations using a local Hugging Face embedding model.
4. **Vector database:** Store the embeddings and associated text in ChromaDB.
5. **Similarity search:** Retrieve relevant chunks based on a user's query.
6. **LLM integration:** Supply the retrieved context to a language model to generate context-aware responses.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | Document processing and RAG orchestration |
| Hugging Face | Local text embedding model |
| ChromaDB | Vector storage and similarity search |
| Google Gemini | Large Language Model integration |
| Streamlit | Web application interface |
| Jupyter Notebook | Experimentation and development |

## 📂 Project Structure

```text
QuizRAG/
├── app.py
├── .gitignore
├── README.md
└── chroma_db/          # Generated locally, not tracked by Git
```

The vector database is excluded from version control. It should be generated locally using the project's document-ingestion pipeline.

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- A Google Gemini API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/VedantAgarwal1905/QuizRAG.git
cd QuizRAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the libraries used by the application and its RAG pipeline:

```bash
pip install streamlit langchain langchain-community langchain-text-splitters langchain-huggingface langchain-google-genai chromadb sentence-transformers pypdf python-dotenv
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or expose API keys in source code.

### 5. Run the application

```bash
streamlit run app.py
```

**Note:** The application may require additional setup depending on how the document-ingestion pipeline and vector database initialization are implemented in the current version.

## 🔒 Security and Data Handling

- API keys are stored in environment variables rather than committed to the repository.
- Local environment files and generated vector database files are excluded using `.gitignore`.
- Users should provide their own API credentials when running the project.

## 🎯 Learning Outcomes

This project explores practical concepts in:

- Retrieval-Augmented Generation (RAG)
- Text preprocessing and recursive chunking
- Dense vector embeddings
- Semantic similarity search
- Vector database integration
- LLM application development
- Building AI applications with Python and Streamlit

## 🔮 Future Improvements

- Interactive quizzes with configurable difficulty levels
- Automated scoring and explanations for answers
- Topic-wise performance analysis
- Identification of weak concepts and personalized revision recommendations
- Support for multiple documents and subjects
- Improved retrieval evaluation and response-quality testing

## 👨‍💻 Author

**Vedant Agarwal**

B.Tech | Artificial Intelligence and Data Science

[GitHub Profile](https://github.com/VedantAgarwal1905)

---

*QuizRAG is a learning project exploring the practical implementation of Retrieval-Augmented Generation for academic revision and self-assessment.*
