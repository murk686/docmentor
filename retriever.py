import os
import json
from datetime import datetime
from fpdf import FPDF
from utils.file_loader import load_document
from utils.chunker import chunk_text
from embedder import embedder, chroma_client, collection
from llm_ollama import synthesize_answer

# ✅ Utility: Strip unsupported characters for PDF
def strip_unsupported(text):
    return ''.join(c for c in text if ord(c) < 256)

# ✅ Load glossary safely
try:
    with open("glossary.json", "r", encoding="utf-8") as f:
        glossary = json.load(f)
except FileNotFoundError:
    print("⚠️ glossary.json not found. Glossary fallback disabled.")
    glossary = {}

# ✅ Model tags
MODEL_TAGS = {
    "Gemma-2B (advanced reasoning)": "gemma:2b",
    "Phi3-mini (smart synthesis)": "phi3",
    "MiniLM (fast retrieval)": "minilm"
}

# ✅ State trackers
query_log = []
file_stats = {}

# ✅ Chunk stats viewer
def show_chunk_stats():
    return "**📊 Chunk Stats:**\n" + "\n".join([f"- `{name}` → {count} chunks" for name, count in file_stats.items()])

# ✅ File ingestion
def ingest_files(files):
    total_chunks = 0
    for file in files:
        text = load_document(file.name)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        print("Sample chunk:", chunks[0])
        total_chunks += len(chunks)
        file_stats[file.name] = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file.name}_{i}"
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{"source": file.name}]
            )

    print("[✓] Collection persisted successfully.")
    print(f"[✓] Embedded {total_chunks} chunks from {len(files)} file(s).")
    return f"✅ Embedded {total_chunks} chunks from {len(files)} file(s)."

# ✅ Persistent query history
HISTORY_FILE = "query_history.json"
query_log = []

# ✅ Load history on startup
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        query_log = json.load(f)

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(query_log, f, indent=2, ensure_ascii=False)

# ✅ Question answering with overview and full context split
def ask_question(query, show_chunks, model_name, show_explanation=False):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    chunks = results['documents'][0]
    sources = results['metadatas'][0]
    distances = results['distances'][0]

    print("🔍 Retrieved chunks:")
    for chunk in chunks:
        print("-", chunk[:100])

    best_score = distances[0] if distances else 1.0
    term = query.lower().strip()
    glossary_hint = glossary.get(term)

    if best_score > 0.85:
        if glossary_hint:
            query_log.append(f"Q: {query}\nA: {glossary_hint} (from glossary)\n---")
            return glossary_hint, "**ℹ️ Answer from glossary.**", "", "\n".join(query_log)
        else:
            query_log.append(f"Q: {query}\nA: No relevant info found.\n---")
            save_history()
            return "No relevant info found in documents or glossary.", "", "", "\n".join(query_log)

    context = "\n\n".join(chunks)
    source_files = ", ".join(set([meta['source'] for meta in sources]))
    model_tag = MODEL_TAGS.get(model_name, "gemma:2b")

    glossary_note = f"\n\nGlossary definition:\n{term}: {glossary_hint}" if glossary_hint else ""
    explanation_note = "\n\nExplain why this answer is correct." if show_explanation else ""
    prompt = f"{query}{glossary_note}\n\nContext:\n{context}{explanation_note}"

    answer = chunks[0] if model_name == "MiniLM (fast retrieval)" else synthesize_answer(prompt, context, model_tag)

    overview = f"**📄 Sources:** {source_files}" if show_chunks else ""
    full_context = f"{overview}\n\n---\n{context}" if show_chunks else ""

    query_log.append(f"Q: {query}\nA: {answer}\n---")
    save_history()
    history_text = "\n".join(query_log)
    return answer, overview, full_context, history_text

# ✅ PDF export with citation footer toggle
def export_answer_to_pdf(question, answer, context, sources=None, model_name=None, include_citations=False):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("DocSmith Answer Export")
    pdf.set_author("DocSmith AI Assistant")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""Generated on: {timestamp}
Model: {model_name}
Sources: {sources}

Question:
{question}

Answer:
{answer}

Context:
{context}
"""
    pdf.multi_cell(0, 10, strip_unsupported(content))

    if include_citations and sources:
        pdf.set_y(-30)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, strip_unsupported(f"Sources: {sources}"))

    pdf.set_y(-20)
    pdf.set_font("Arial", size=8)
    pdf.cell(0, 10, strip_unsupported("Exported by DocSmith • Powered by offline synthesis"), align="C")

    filename = f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.abspath(filename)
    pdf.output(filepath)
    print(f"[✓] PDF saved to: {filepath}")
    print("[✓] File exists:", os.path.exists(filepath))
    return filepath

# ✅ TXT export with optional citation footer
def save_answer_to_file(answer, sources=None):
    filename = f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.abspath(filename)
    if sources:
        answer += f"\n\nSources: {sources}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(answer)
    print(f"[✓] TXT saved to: {filepath}")
    return filepath

# ✅ Suggested questions with style and glossary awareness
def suggest_questions(style="Insightful"):
    results = collection.query(query_embeddings=[[0.0]*384], n_results=5)
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    context = "\n".join(chunks)
    source_files = ", ".join(set([meta['source'] for meta in metadatas]))

    glossary_terms = [term for term in glossary if term in context.lower()]
    glossary_hint = f"\n\nGlossary terms in context: {', '.join(glossary_terms)}" if glossary_terms else ""

    style_map = {
        "Insightful": "Suggest 3 insightful questions a student might ask.",
        "Factual": "Suggest 3 factual questions based on the content.",
        "Multiple Choice": "Create 3 multiple-choice questions with answers.",
        "Open-ended": "Generate 3 open-ended discussion questions."
    }
    instruction = style_map.get(style, style_map["Insightful"])

    prompt = f"""Based on the following document excerpts:{glossary_hint}

{context}

{instruction}
"""
    raw = synthesize_answer(prompt, context, MODEL_TAGS["Gemma-2B (advanced reasoning)"])
    seen = set()
    questions = []
    for q in raw.split("\n"):
        cleaned = q.strip("-•1234567890. ").strip()
        if cleaned and cleaned not in seen:
            questions.append(cleaned)
            seen.add(cleaned)

    return questions, f"**Sources:** {source_files}"