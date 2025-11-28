from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from src.search import RAGSearch
import hashlib
import json
import datetime
from PyPDF2 import PdfReader
import numpy as np
from src.embedding import EmbeddingPipeline

app = Flask(__name__, static_folder="build", static_url_path="")

UPLOAD_FOLDER = "data"
PROCESSED_FOLDER = "processed_data"
PROCESSED_INDEX = os.path.join(PROCESSED_FOLDER, "processed_index.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

if not os.path.exists(PROCESSED_INDEX):
    with open(PROCESSED_INDEX, "w", encoding="utf-8") as f:
        json.dump([], f)

rag = None

def get_rag():
    global rag
    if rag is None:
        print("Initializing RAGSearch...")
        rag = RAGSearch()
    return rag

def compute_file_hash(path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def extract_text_from_pdf(path) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            text_parts.append(txt)
    except Exception as e:
        print(f" PDF extraction error for {path}: {e}")
    return "\n\n".join(text_parts).strip()

def load_processed_index():
    with open(PROCESSED_INDEX, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def save_processed_index(index_list):
    with open(PROCESSED_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_list, f, indent=2, ensure_ascii=False)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join("build", path)):
        return send_from_directory("build", path)
    return send_from_directory("build", "index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    print("\n /upload hit")
    print("➡ request.files keys:", list(request.files.keys()))

    if "file" not in request.files:
        print(" ERROR: No 'file' in request.files")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        print(" ERROR: No filename provided")
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(temp_path)
    print(f"Saved uploaded file temporarily to {temp_path}")

    file_hash = compute_file_hash(temp_path)
    print(f"File hash: {file_hash}")

    processed_index = load_processed_index()
    for rec in processed_index:
        if rec.get("hash") == file_hash:
            try:
                os.remove(temp_path)
            except Exception:
                pass
            print(f"Duplicate detected (hash match). Skipping processing.")
            return jsonify({"message": "File already processed earlier. Ready to chat!"})

    print(" Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(temp_path)
    if not pdf_text:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return jsonify({"error": "Failed to extract text from PDF"}), 500

    class SimpleDoc:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    doc_obj = SimpleDoc(pdf_text, metadata={"source_filename": filename})

    print("Chunking & embedding the new PDF (incremental)...")
    emb_pipe = EmbeddingPipeline(model_name="all-MiniLM-L6-v2") 
    chunks = emb_pipe.chunk_documents([doc_obj])
    embeddings = emb_pipe.embed_chunks(chunks)  
    print(f"Generated embeddings shape: ({len(embeddings)}, ?)")

    arr = np.array(embeddings).astype("float32")

    rag_instance = get_rag()

    if rag_instance.vectorstore.index is None:
        print("FAISS index doesn't exist yet. Creating new index for incremental add.")
        rag_instance.vectorstore.add_embeddings(arr, [{"text": c.page_content} for c in chunks])
        rag_instance.vectorstore.save()
    else:
        rag_instance.vectorstore.add_embeddings(arr, [{"text": c.page_content} for c in chunks])
        rag_instance.vectorstore.save()

    dest_name = filename
    dest_path = os.path.join(PROCESSED_FOLDER, dest_name)
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(dest_name)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        dest_name = f"{base}_{ts}{ext}"
        dest_path = os.path.join(PROCESSED_FOLDER, dest_name)

    os.rename(temp_path, dest_path)
    print(f" Moved file to processed_data: {dest_path}")

    new_rec = {
        "filename": dest_name,
        "hash": file_hash,
        "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "path": dest_path
    }
    processed_index.append(new_rec)
    save_processed_index(processed_index)
    print("Updated processed_index.json")

    return jsonify({"message": "File uploaded, processed, and stored successfully!", "file": new_rec})

@app.route("/processed", methods=["GET"])
def list_processed():
    processed_index = load_processed_index()
    processed_index_sorted = sorted(processed_index, key=lambda x: x.get("processed_at", ""), reverse=True)
    return jsonify({"processed": processed_index_sorted})

@app.route("/chat", methods=["POST"])
def chat():
    rag_instance = get_rag()
    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    print(f" User asked: {user_message}")
    response = rag_instance.search_and_summarize(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    print("Starting AgriBot Server...")
    app.run(host="0.0.0.0", port=5000)
