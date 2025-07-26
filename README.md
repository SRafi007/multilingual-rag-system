# Multilingual RAG System

A simple Retrieval-Augmented Generation (RAG) system that supports both English and Bengali queries, designed to fetch relevant information from PDF documents and generate meaningful answers.

## 🎯 Objective

Develop a basic RAG pipeline capable of understanding and responding to both English and Bengali queries by retrieving relevant information from a PDF document corpus and generating grounded answers.

## ✨ Features

### Core Functionality
- **Multilingual Support**: Accepts user queries in both English and Bengali
- **Document Retrieval**: Fetches relevant document chunks from knowledge base
- **Contextual Answers**: Generates responses based on retrieved information
- **Memory Management**: Maintains both short-term and long-term memory

### Knowledge Base
- **Source**: HSC26 Bangla 1st Paper (Bengali textbook)
- **Processing**: Advanced pre-processing and data cleaning for improved chunk accuracy
- **Storage**: Document chunking and vectorization in vector database

### Memory System
- **Short-Term Memory**: Recent chat sequence inputs
- **Long-Term Memory**: PDF document corpus stored in vector database

## 🚀 API Endpoints

### Conversation API
```
POST /chat
```
**Request Body:**
```json
{
  "query": "অনুপমের ভাষায় সত্যপুরুষ কাকে বলা হয়েছে?"
}
```

**Response:**
```json
{
  "answer": "শুভনাথ",
  "sources": ["chunk_id_1", "chunk_id_2"],
  "confidence": 0.85
}
```

## 📋 Sample Test Cases

| Query (Bengali) | Expected Answer |
|----------------|-----------------|
| অনুপমের ভাষায় সত্যপুরুষ কাকে বলা হয়েছে? | শুভনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd multilingual-rag-system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## 📖 Usage

### Command Line Interface
```bash
python main.py --query "অনুপমের ভাষায় সত্যপুরুষ কাকে বলা হয়েছে?"
```

### REST API
```bash
# Start the server
python app.py

# Make requests
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"}'
```
