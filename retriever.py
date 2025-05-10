import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def retrieve(query):
    try:
        model_path = os.path.abspath("./models/paraphrase-MiniLM-L3-v2")
        embeddings = HuggingFaceEmbeddings(model_name=model_path)

        # Load FAISS vector store
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

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
