import re

print("[✓] chunker.py loaded successfully")

def chunk_text(text, max_length=2000, overlap=200):
    """
    Splits text into overlapping chunks based on line accumulation.
    Each chunk will be under max_length characters, with overlap preserved.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    lines = text.split('. ')  # Split by sentence for better semantic grouping
    chunks = []
    i = 0

    while i < len(lines):
        current = ""
        start = i
        while i < len(lines) and len(current) + len(lines[i]) < max_length:
            current += lines[i].strip() + ". "
            i += 1
        chunks.append(current.strip())
        i = max(start + 1, i - overlap)  # Backtrack for overlap

    return chunks

# Debug mode (optional)
if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    print(f"[✓] Generated {len(chunks)} chunks.")
    print("🔍 Sample chunk:\n", chunks[0])