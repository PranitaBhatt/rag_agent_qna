import os
import json
import requests
from dotenv import load_dotenv
from retriever import retrieve
from tools import calculator_tool, dictionary_tool

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("QNA_APIKEY")  # This must be below load_dotenv()

print("[DEBUG] QNA_APIKEY loaded:", api_key)

# Ensure the key is loaded
if not api_key:
    raise ValueError("API key not found. Make sure QNA_APIKEY is set in the .env file.")

# Groq API details
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"  # Confirmed working model

# API headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def generate_answer(query):
    # Built-in calculator
    if "calculate" in query.lower():
        expr = query.lower().replace("calculate", "").strip()
        return f"[Calculator Tool]\nResult: {calculator_tool(expr)}"

    # Built-in dictionary
    elif "define" in query.lower():
        word = query.lower().replace("define", "").strip()
        return f"[Dictionary Tool]\n{dictionary_tool(word)}"

    # Default: RAG-powered retrieval + LLM
    else:
        context = retrieve(query)

        if not context or "[Retriever failed" in context[0]:
            return "[Error] Failed to retrieve context. Check model path or FAISS index."

        prompt = f"Context:\n{''.join(context)}\n\nQuestion: {query}\nAnswer:"

        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 256
            }

            print("Sending request to Groq API...")
            print("Payload:\n", json.dumps(payload, indent=2))

            response = requests.post(API_URL, headers=headers, json=payload)

            if response.status_code != 200:
                print("[Groq API Error] Status Code:", response.status_code)
                print("Response Text:", response.text)
                return f"[Error] Groq API responded with {response.status_code}"

            data = response.json()
            return f"[RAG Response]\n{data['choices'][0]['message']['content']}"

        except Exception as e:
            return f"[Error]\n{str(e)}"

# Optional: Debug test
if __name__ == "__main__":
    test_query = "What does the company do?"
    print("Testing generate_answer:")
    print(generate_answer(test_query))
