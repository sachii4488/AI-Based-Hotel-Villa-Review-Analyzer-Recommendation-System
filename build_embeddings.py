"""
Build Embeddings script for Hotel & Villa Review Chatbot (RAG).

Loads output/reviews_by_hotel.json, embeds each review's text using
SentenceTransformer("all-MiniLM-L6-v2"), and stores everything in a
persistent ChromaDB collection at output/chroma_db.

This is a ONE-TIME setup step. Run it once; the chatbot script will
later query this same database.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_JSON = "output/reviews_by_hotel.json"
CHROMA_PATH = "output/chroma_db"
COLLECTION_NAME = "reviews_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64  # how many reviews to embed/add at once


def main():
    print(f"Loading {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        reviews_by_hotel = json.load(f)

    total_reviews = sum(len(reviews) for reviews in reviews_by_hotel.values())
    print(f"Loaded {len(reviews_by_hotel)} hotels, {total_reviews} reviews total")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}'...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Initializing ChromaDB at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # If the collection already exists from a previous run, delete it
    # so we start fresh (avoids duplicate entries).
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — deleting and recreating...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    # Build flat lists: one entry per review
    ids = []
    documents = []
    metadatas = []

    for hotel_id, reviews in reviews_by_hotel.items():
        for review in reviews:
            text = review.get("text")
            if not text or not str(text).strip():
                continue

            review_id = str(review.get("review_id"))
            # Chroma requires globally unique IDs -> combine hotel_id + review_id
            unique_id = f"{hotel_id}_{review_id}"

            ids.append(unique_id)
            documents.append(str(text))
            metadatas.append({
                "hotel_id": str(hotel_id),
                "review_id": review_id,
                "rating": str(review.get("rating", "")),
            })

    print(f"Prepared {len(documents)} reviews for embedding")

    # Process in batches
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )

        print(f"  Added batch {start}-{min(end, len(documents))} / {len(documents)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    print(f"Persistent DB saved at: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
