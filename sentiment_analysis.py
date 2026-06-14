"""
Sentiment Analysis script for Hotel & Villa Review Analyzer.

Loads output/reviews_df.csv, runs each review through a pretrained
sentiment classifier, and saves:
  - output/reviews_with_sentiment.csv  (full table + sentiment columns)
  - output/hotel_sentiment_summary.json  ({hotel_id: {positive_pct, negative_pct, avg_score, ...}})
"""

import os
import json
import pandas as pd
from transformers import pipeline

INPUT_CSV = "output/reviews_df.csv"
OUTPUT_CSV = "output/reviews_with_sentiment.csv"
OUTPUT_JSON = "output/hotel_sentiment_summary.json"


def run_sentiment_analysis(df):
    print("Loading sentiment model (first run will download ~260MB)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    texts = df["text"].astype(str).tolist()

    print(f"Running sentiment analysis on {len(texts)} reviews...")
    results = sentiment_pipeline(texts, truncation=True, batch_size=16)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    return df


def build_hotel_summary(df):
    summary = {}

    for hotel_id, group in df.groupby("hotel_id"):
        total = len(group)
        positive = (group["sentiment_label"] == "POSITIVE").sum()
        negative = (group["sentiment_label"] == "NEGATIVE").sum()

        summary[str(hotel_id)] = {
            "hotel_name": group["hotel_name"].iloc[0],
            "city": group["city"].iloc[0],
            "total_reviews_analyzed": int(total),
            "positive_count": int(positive),
            "negative_count": int(negative),
            "positive_pct": round(positive / total * 100, 1),
            "negative_pct": round(negative / total * 100, 1),
            "avg_sentiment_score": round(float(group["sentiment_score"].mean()), 3),
            "avg_star_rating": round(float(pd.to_numeric(group["rating"], errors="coerce").mean()), 2),
        }

    return summary


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} reviews")

    df = run_sentiment_analysis(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    summary = build_hotel_summary(df)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")

    # Print a few examples
    print("\nSample hotel summaries:")
    for hotel_id, s in list(summary.items())[:3]:
        print(f"  {s['hotel_name']} ({s['city']}): "
              f"{s['positive_pct']}% positive, "
              f"avg star rating {s['avg_star_rating']}, "
              f"{s['total_reviews_analyzed']} reviews")


if __name__ == "__main__":
    main()
