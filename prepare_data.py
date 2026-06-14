"""
Data preparation script for Hotel & Villa Review Analyzer & Recommendation System.

Loads all CSVs in DATA_DIR and builds:
  - reviews_df.csv
  - hotels_lookup.json
  - reviews_by_hotel.json
"""

import os
import json
import pandas as pd

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KEEP_COLS = [
    "id",
    "hotel_id",
    "hotel_name",
    "district",
    "text",
    "rating",
    "subratings",
    "publishedDate",
    "tripType",
    "hotel_overall_rating",
    
    "latitude",
    "longitude",
    "num_reviews_total"
]


def load_all_csvs(data_dir):
    all_dfs = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".csv"):
            continue

        fpath = os.path.join(data_dir, fname)

        print(f"Loading {fname}...")

        df = pd.read_csv(
            fpath,
            usecols=lambda c: c in KEEP_COLS,
            low_memory=False
        )

        # Drop rows with missing essential values
        df = df.dropna(subset=["hotel_id", "hotel_name", "text"])

        # Use district as city
        df["city"] = df["district"]

        df["source_file"] = fname

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No CSV files found in data folder.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates
    combined = combined.drop_duplicates(subset=["id"], keep="first")

    return combined


def build_hotels_lookup(df):
    lookup = {}

    hotel_cols = [
        "city",
        "hotel_id",
        "hotel_name",
        "hotel_overall_rating",
        "latitude",
        "longitude",
        "num_reviews_total"
    ]

    hotels = df[hotel_cols].drop_duplicates(subset=["hotel_id"])

    for city, group in hotels.groupby("city"):
        lookup[city] = []

        for _, row in group.iterrows():
            lookup[city].append({
                "hotel_id": str(row["hotel_id"]),
                "name": row["hotel_name"],
                "rating": row["hotel_overall_rating"],
                "num_reviews": row["num_reviews_total"],
                "latitude": row["latitude"],
                "longitude": row["longitude"]
            })

    return lookup


def build_reviews_by_hotel(df):
    reviews_by_hotel = {}

    for hotel_id, group in df.groupby("hotel_id"):

        reviews = []

        for _, row in group.iterrows():
            reviews.append({
                "review_id": str(row["id"]),
                "text": row["text"],
                "rating": row["rating"],
                "subratings": row.get("subratings"),
                "date": row.get("publishedDate"),
                "trip_type": row.get("tripType")
            })

        reviews_by_hotel[str(hotel_id)] = reviews

    return reviews_by_hotel


def main():
    print("Loading and combining CSVs...")

    df = load_all_csvs(DATA_DIR)

    print(f"Total reviews after cleaning/dedup: {len(df)}")
    print(f"Total unique hotels: {df['hotel_id'].nunique()}")
    print(f"Total cities: {df['city'].nunique()}")
    print("Cities found:", sorted(df["city"].dropna().unique().tolist()))

    # Save reviews dataframe
    reviews_out = os.path.join(OUTPUT_DIR, "reviews_df.csv")

    flat_cols = [
        "id",
        "hotel_id",
        "hotel_name",
        "city",
        "text",
        "rating",
        "subratings",
        "publishedDate",
        "tripType",
        "hotel_overall_rating",
        "latitude",
        "longitude",
        "num_reviews_total"
    ]

    df[flat_cols].to_csv(reviews_out, index=False)
    print(f"Saved {reviews_out}")

    # Save hotel lookup
    hotels_lookup = build_hotels_lookup(df)

    hotels_out = os.path.join(OUTPUT_DIR, "hotels_lookup.json")

    with open(hotels_out, "w", encoding="utf-8") as f:
        json.dump(hotels_lookup, f, indent=2, default=str)

    print(f"Saved {hotels_out}")

    # Save reviews by hotel
    reviews_by_hotel = build_reviews_by_hotel(df)

    reviews_by_hotel_out = os.path.join(
        OUTPUT_DIR,
        "reviews_by_hotel.json"
    )

    with open(reviews_by_hotel_out, "w", encoding="utf-8") as f:
        json.dump(reviews_by_hotel, f, indent=2, default=str)

    print(f"Saved {reviews_by_hotel_out}")


if __name__ == "__main__":
    main()