from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(folder_path="documents"):
    docs = []
    for fname in os.listdir(folder_path):
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(docs)
