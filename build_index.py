import os
from utils import load_documents, chunk_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_index():
    # Resolve base directory relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load and chunk documents
    docs = load_documents()
    chunks = chunk_documents(docs)

    # Use robust path to the model
    model_path = os.path.join(BASE_DIR, "models", "paraphrase-MiniLM-L3-v2")

    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    # Build and save FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(os.path.join(BASE_DIR, "vector_store"))
    print("✅ Vector index built and saved to 'vector_store/'")

if __name__ == "__main__":
    build_index()
