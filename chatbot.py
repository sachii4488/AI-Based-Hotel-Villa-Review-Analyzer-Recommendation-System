"""
RAG Chatbot for Hotel & Villa Review Analyzer.

Given a hotel_id and a user question:
  1. Embed the question using the same SentenceTransformer used for indexing
  2. Query ChromaDB (filtered to that hotel) for the top 5 most relevant reviews
  3. Build a grounded prompt using only those reviews
  4. Send the prompt to Claude and return the answer
"""

import json
import os
from difflib import SequenceMatcher
from functools import lru_cache
from urllib import error, parse, request

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

CHROMA_PATH = "output/chroma_db"
COLLECTION_NAME = "reviews_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
HOTELS_LOOKUP_PATH = "output/hotels_lookup.json"
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_PLACES_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Load these once at module level so they're reused across calls
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

llm_client = Groq()  # reads GROQ_API_KEY from environment


@lru_cache(maxsize=1)
def load_hotels_lookup():
    if not os.path.exists(HOTELS_LOOKUP_PATH):
        return {}

    with open(HOTELS_LOOKUP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_hotel_context(hotel_id):
    lookup = load_hotels_lookup()

    for city, hotels in lookup.items():
        for hotel in hotels:
            if str(hotel.get("hotel_id")) == str(hotel_id):
                return {
                    "hotel_id": str(hotel_id),
                    "name": hotel.get("name", ""),
                    "city": city,
                    "latitude": hotel.get("latitude"),
                    "longitude": hotel.get("longitude"),
                }

    return None


def needs_factual_lookup(question):
    q = question.lower()
    factual_keywords = [
        "location",
        "address",
        "where",
        "phone",
        "contact",
        "website",
        "email",
        "map",
        "directions",
        "near",
        "how far",
    ]
    return any(keyword in q for keyword in factual_keywords)


def needs_review_context(question):
    q = question.lower()
    review_keywords = [
        "review",
        "reviews",
        "guests say",
        "what do guests",
        "feedback",
        "breakfast",
        "pool",
        "service",
        "clean",
        "cleanliness",
        "wifi",
        "parking",
        "staff",
        "food",
    ]
    return any(keyword in q for keyword in review_keywords)


def http_get_json(url, params, timeout=10):
    query = parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})

    with request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload)


def names_are_similar(name_a, name_b, threshold=0.4):
    """
    Return True if two hotel names are similar enough to be considered a match.
    Uses SequenceMatcher for character-level fuzzy comparison.
    A threshold of 0.4 is lenient enough to handle abbreviations, punctuation
    differences, and extra words like '& Spa' or 'by XYZ'.
    """
    ratio = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
    return ratio >= threshold


def lookup_google_places(hotel):
    if not GOOGLE_PLACES_API_KEY:
        return {
            "ok": False,
            "message": (
                "I could not verify the address/phone because GOOGLE_PLACES_API_KEY is not configured."
            ),
        }

    query = f"{hotel['name']}, {hotel.get('city', '')}, Sri Lanka"

    try:
        search_data = http_get_json(
            GOOGLE_PLACES_TEXTSEARCH_URL,
            {
                "query": query,
                "key": GOOGLE_PLACES_API_KEY,
            },
        )

        if search_data.get("status") != "OK" or not search_data.get("results"):
            return {
                "ok": False,
                "message": "I could not verify the address/phone from Google Places.",
            }

        candidate = search_data["results"][0]
        candidate_name = candidate.get("name", "")
        hotel_name = hotel.get("name", "")

        # Fuzzy name match — tolerates abbreviations, punctuation, extra words
        if hotel_name and candidate_name:
            if not names_are_similar(hotel_name, candidate_name, threshold=0.4):
                return {
                    "ok": False,
                    "message": (
                        "I could not verify the address/phone because Google Places "
                        "returned an uncertain match."
                    ),
                }

        place_id = candidate.get("place_id")
        if not place_id:
            return {
                "ok": False,
                "message": "I could not verify the address/phone from Google Places.",
            }

        details_data = http_get_json(
            GOOGLE_PLACES_DETAILS_URL,
            {
                "place_id": place_id,
                "fields": "name,formatted_address,international_phone_number,website,url,geometry",
                "key": GOOGLE_PLACES_API_KEY,
            },
        )

        if details_data.get("status") != "OK":
            return {
                "ok": False,
                "message": "I could not verify the address/phone from Google Places.",
            }

        place = details_data.get("result", {})
        location = place.get("geometry", {}).get("location", {})

        return {
            "ok": True,
            "name": place.get("name", hotel.get("name", "")),
            "address": place.get("formatted_address"),
            "phone": place.get("international_phone_number"),
            "website": place.get("website"),
            "maps_url": place.get("url"),
            "lat": location.get("lat"),
            "lng": location.get("lng"),
        }

    except (error.URLError, TimeoutError, ValueError, KeyError):
        return {
            "ok": False,
            "message": "I could not verify the address/phone from Google Places.",
        }


def format_factual_answer(place_info, question=""):
    if not place_info.get("ok"):
        return place_info.get("message", "I could not verify the address/phone.")

    q = question.lower()
    lines = [f"Here is the information for {place_info['name']}:"]

    # Address / location / directions
    if any(k in q for k in ["location", "address", "where", "directions", "map", "near", "how far"]):
        if place_info.get("address"):
            lines.append(f"Address: {place_info['address']}")
        if place_info.get("maps_url"):
            lines.append(f"Google Maps: {place_info['maps_url']}")

    # Phone / contact
    if any(k in q for k in ["phone", "contact", "call", "number"]):
        if place_info.get("phone"):
            lines.append(f"Phone: {place_info['phone']}")
        else:
            lines.append("Phone: I could not verify the phone number.")

    # Website / email
    if any(k in q for k in ["website", "email", "online", "web"]):
        if place_info.get("website"):
            lines.append(f"Website: {place_info['website']}")
        else:
            lines.append("Website: I could not verify the website.")

    # Fallback — if nothing matched, return everything
    if len(lines) == 1:
        if place_info.get("address"):
            lines.append(f"Address: {place_info['address']}")
        if place_info.get("phone"):
            lines.append(f"Phone: {place_info['phone']}")
        if place_info.get("website"):
            lines.append(f"Website: {place_info['website']}")
        if place_info.get("maps_url"):
            lines.append(f"Google Maps: {place_info['maps_url']}")

    return "\n".join(lines)


def answer_review_question(hotel_id, question):
    hotel_id = str(hotel_id)

    query_embedding = embed_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        where={"hotel_id": hotel_id},
    )

    retrieved_docs = results.get("documents", [[]])[0]

    if not retrieved_docs:
        return "I don't have any reviews for this hotel to answer that question."

    reviews_text = "\n\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = (
        f"Reviews:\n{reviews_text}\n\n"
        f"Question: {question}\n"
        f"Answer using only the above reviews. If not enough info, say so."
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def answer_question(hotel_id, question):
    hotel_id = str(hotel_id)
    hotel = get_hotel_context(hotel_id)

    factual_needed = needs_factual_lookup(question)
    review_needed = needs_review_context(question) or not factual_needed

    factual_text = None
    if factual_needed:
        if hotel is None:
            factual_text = (
                "I could not verify the address/phone because this hotel is not in the local hotel index."
            )
        else:
            factual_text = format_factual_answer(lookup_google_places(hotel), question)

    review_text = None
    if review_needed:
        review_text = answer_review_question(hotel_id, question)

    if factual_needed and review_needed:
        return f"{factual_text}\n\nReview-based context:\n{review_text}"

    if factual_needed:
        return factual_text

    return review_text


if __name__ == "__main__":
    # Manual test
    test_hotel_id = "8073048"
    test_question = "What do guests say about the breakfast?"

    print(f"\nHotel ID: {test_hotel_id}")
    print(f"Question: {test_question}\n")

    answer = answer_question(test_hotel_id, test_question)
    print(f"Answer:\n{answer}")