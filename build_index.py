from utils import load_documents, chunk_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def build_index():
    # Load and chunk documents
    docs = load_documents()
    chunks = chunk_documents(docs)

    # Use absolute path with raw string or forward slashes
    model_path = os.path.abspath("./models/paraphrase-MiniLM-L3-v2")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path
    )

    # Build and save FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vector_store")
    print("✅ Vector index built and saved to 'vector_store/'")

if __name__ == "__main__":
    build_index()
