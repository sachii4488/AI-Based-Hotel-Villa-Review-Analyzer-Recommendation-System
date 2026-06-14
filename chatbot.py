"""
RAG Chatbot for Hotel & Villa Review Analyzer.

Given a hotel_id and a user question:
  1. Embed the question using the same SentenceTransformer used for indexing
  2. Query ChromaDB (filtered to that hotel) for the top 5 most relevant reviews
  3. Build a grounded prompt using only those reviews
  4. Send the prompt to Claude and return the answer
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_PATH = "output/chroma_db"
COLLECTION_NAME = "reviews_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# Load these once at module level so they're reused across calls
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

llm_client = Groq()  # reads GROQ_API_KEY from environment


def answer_question(hotel_id, question):
    hotel_id = str(hotel_id)

    # 1. Embed the question
    query_embedding = embed_model.encode([question]).tolist()

    # 2. Query ChromaDB, filtered to this hotel, top 5 results
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        where={"hotel_id": hotel_id},
    )

    retrieved_docs = results.get("documents", [[]])[0]

    if not retrieved_docs:
        return "I don't have any reviews for this hotel to answer that question."

    # 3. Build the grounded prompt
    reviews_text = "\n\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = (
        f"Reviews:\n{reviews_text}\n\n"
        f"Question: {question}\n"
        f"Answer using only the above reviews. If not enough info, say so."
    )

    # 4. Send to LLM
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Manual test
    test_hotel_id = "8073048"
    test_question = "What do guests say about the breakfast?"

    print(f"\nHotel ID: {test_hotel_id}")
    print(f"Question: {test_question}\n")

    answer = answer_question(test_hotel_id, test_question)
    print(f"Answer:\n{answer}")
