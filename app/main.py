"""
FastAPI backend for Hotel & Villa Review Analyzer & Recommendation System.

Endpoints:
  GET  /cities                       -> list of cities
  GET  /cities/{city}/hotels         -> hotels in that city
  GET  /hotels/{hotel_id}/summary    -> sentiment + pros/cons for that hotel
  POST /hotels/{hotel_id}/chat       -> {"question": "..."} -> {"answer": "..."}
"""

import json
import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing chatbot.py from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chatbot import answer_question  # noqa: E402

app = FastAPI(title="Hotel & Villa Review Analyzer API")

# Allow frontend (e.g. React on localhost:3000 or Streamlit) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def load_json(filename, default=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using empty default")
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load all data into memory at startup
hotels_lookup = load_json("hotels_lookup.json", default={})
hotel_sentiment_summary = load_json("hotel_sentiment_summary.json", default={})
hotel_pros_cons = load_json("hotel_pros_cons.json", default={})

print(f"Loaded {len(hotels_lookup)} cities")
print(f"Loaded sentiment for {len(hotel_sentiment_summary)} hotels")
print(f"Loaded pros/cons for {len(hotel_pros_cons)} hotels")


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Hotel & Villa Review Analyzer API is running"}


@app.get("/cities")
def get_cities():
    return list(hotels_lookup.keys())


@app.get("/cities/{city}/hotels")
def get_hotels_by_city(city: str):
    if city not in hotels_lookup:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    return hotels_lookup[city]


@app.get("/hotels/{hotel_id}/summary")
def get_hotel_summary(hotel_id: str):
    sentiment = hotel_sentiment_summary.get(hotel_id)
    pros_cons = hotel_pros_cons.get(hotel_id, {"pros": [], "cons": []})

    if sentiment is None and hotel_id not in hotel_pros_cons:
        raise HTTPException(status_code=404, detail=f"Hotel '{hotel_id}' not found")

    return {
        "hotel_id": hotel_id,
        "sentiment": sentiment or {},
        "pros": pros_cons.get("pros", []),
        "cons": pros_cons.get("cons", []),
    }


@app.post("/hotels/{hotel_id}/chat")
def chat_with_hotel(hotel_id: str, request: ChatRequest):
    answer = answer_question(hotel_id, request.question)
    return {"answer": answer}
