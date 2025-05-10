import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def retrieve(query):
    try:
        # Resolve base directory relative to this script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Construct model and FAISS index paths
        model_path = os.path.join(BASE_DIR, "models", "paraphrase-MiniLM-L3-v2")
        vector_store_path = os.path.join(BASE_DIR, "vector_store")

        # Debug: Optional print to confirm paths
        if not os.path.exists(model_path):
            print(f"❌ Model path does not exist: {model_path}")
        if not os.path.exists(vector_store_path):
            print(f"❌ Vector store path does not exist: {vector_store_path}")

        embeddings = HuggingFaceEmbeddings(model_name=model_path)

        # Load FAISS vector store
        db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = db.similarity_search(query, k=3)

        # Extract and clean content
        cleaned_context = []
        for doc in docs:
            if isinstance(doc, Document):
                content = doc.page_content.strip().replace("\n", " ")
                cleaned_context.append(content)

        return cleaned_context

    except Exception as e:
        print(f"[Retriever Error] {str(e)}")
        return ["[Retriever failed to fetch context]"]
