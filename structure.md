```
multilingual_rag_system/
│
├── 📁 app/                                 # Main application logic
│   ├── 📁 api/                             # REST API endpoints
│   │   ├── __init__.py
│   │   └── rag_endpoint.py                # Accepts user input, returns responses
│   │
│   ├── 📁 data_pipeline/                        # Core data flow modules
│   │   ├── __init__.py
│   │   ├── pdf_reader.py                  # PDF to image/text
│   │   ├── ocr_engine.py                  # OCR if scanned PDF
│   │   ├── preprocessor.py                # Text cleaning, normalization
│   │   ├── chunker.py                     # Chunk text into manageable units
│   │   └── embedder.py                    # Convert chunks to embeddings
│   │
│   ├── 📁 vector_store/                   # Vector DB management
│   │   ├── __init__.py
│   │   ├── faiss_handler.py               # Faiss integration
│   │   ├── index_builder.py               # Builds/updates the index
│   │   └── retriever.py                   # Finds relevant chunks
│   │
│   ├── 📁 models/                         # Language models & generation
│   │   ├── __init__.py
│   │   ├── rag_model.py                   # RAG logic (retrieval + generation)
│   │   └── translator.py                  # Bangla ↔ English translation if needed
│   │
│   ├── 📁 memory/                         # Memory management
│   │   ├── __init__.py
│   │   ├── short_term.py                  # Handles conversational context
│   │   └── long_term.py                   # Vector DB as long-term memory
│   │
│   ├── 📁 agents/                         # Optional: agent logic for chaining, tools
│   │   ├── __init__.py
│   │   └── simple_agent.py
│   │
│   ├── 📁 data/                           # Raw and processed data
│   │   ├── 📁 raw/                         # Original HSC26 PDF files
│   │   ├── 📁 processed/                   # Cleaned & chunked text
│   │   └── 📁 embeddings/                  # Saved embeddings
│   │
│   └── main.py                           # Entry point (FastAPI / CLI)
│
├── 📁 config/                             # Config and constants
│   ├── __init__.py
│   └── settings.py                        # API keys, file paths, model configs
│
├── 📁 tests/                              # Unit and integration tests
│   ├── __init__.py
│   ├── test_retriever.py
│   ├── test_embedder.py
│   └── test_api.py
│
├── 📁 notebooks/                          # Jupyter/Colab for experiments
│   └── exploration.ipynb
│
├── .env                                   # Environment variables
├── requirements.txt                       # Python dependencies
├── README.md                              # Project overview
└── vercel.json / Dockerfile               # Deployment configs
```