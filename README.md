# 📚 DocSmith — Modular Document Q&A Assistant

DocSmith is a local, offline-friendly document question-answering system built with Python, Gradio, and ChromaDB. It allows users to upload `.txt`, `.pdf`, or `.docx` files, embed their contents, and ask natural language questions with context-aware answers.

---

## 🚀 Features

- ✅ Upload multiple documents
- ✅ Chunk and embed content using SentenceTransformers
- ✅ Ask questions and retrieve relevant context
- ✅ Toggle visibility of retrieved chunks
- ✅ Markdown-formatted output with source attribution
- ✅ Modular architecture for easy scaling

---

## 🧠 Architecture

---

## ⚙️ Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/docsmith
   cd docsmith

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies

pip install -r requirements.txt

# Run the app
python run app.py

📂 Supported File Types
- .txt
- .pdf (text-based)
- .docx

🧩 How It Works
- Ingestion: Uploaded files are read, chunked, and embedded into ChromaDB.
- Querying: Questions are encoded and matched against stored chunks.
- Answering: Top chunks are returned with source info and optional context.

👨‍🏫 Built By
Murk — Computer Science graduate, educational mentor, and technologist.
Silver Medalist | Python & Web Developer | System Optimizer | AI Integrator

📄 License
MIT License — free to use, modify, and distribute.

---
