"""
Pros & Cons Generation script for Hotel & Villa Review Analyzer.

For each hotel:
  - Splits reviews into positive (rating >= 4) and negative (rating <= 2)
  - Takes top N reviews from each group
  - Sends them to an LLM (Claude API) with a prompt asking for recurring
    pros / cons
  - Saves output/hotel_pros_cons.json -> {hotel_id: {pros: [...], cons: [...]}}

This is precomputed ONCE for all hotels (not called live by the app).
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

INPUT_CSV = "output/reviews_with_sentiment.csv"  # fallback to reviews_df.csv if this doesn't exist
OUTPUT_JSON = "output/hotel_pros_cons.json"

MAX_REVIEWS_PER_GROUP = 12   # take up to N positive / N negative reviews per hotel
MODEL = "llama-3.3-70b-versatile"

client = Groq()  # reads GROQ_API_KEY from environment


def build_prompt(hotel_name, positive_reviews, negative_reviews):
    pos_text = "\n".join(f"- {r}" for r in positive_reviews) if positive_reviews else "(none available)"
    neg_text = "\n".join(f"- {r}" for r in negative_reviews) if negative_reviews else "(none available)"

    prompt = f"""You are analyzing guest reviews for the hotel/villa "{hotel_name}".

POSITIVE REVIEWS:
{pos_text}

NEGATIVE REVIEWS:
{neg_text}

Based ONLY on the reviews above, respond with valid JSON in exactly this format
(no extra text, no markdown fences):

{{
  "pros": ["short point 1", "short point 2", "short point 3"],
  "cons": ["short point 1", "short point 2", "short point 3"]
}}

List 3-5 recurring points for each. If a category has no reviews or nothing
clear emerges, return an empty list for that category. Keep each point under
15 words.
"""
    return prompt


def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()

            # Strip markdown fences if the model added them anyway
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            return json.loads(text)
        except json.JSONDecodeError:
            print(f"  Warning: could not parse JSON response, retrying... (attempt {attempt+1})")
            time.sleep(1)
        except Exception as e:
            print(f"  Error calling LLM: {e}, retrying... (attempt {attempt+1})")
            time.sleep(2)

    # Fallback if all retries fail
    return {"pros": [], "cons": []}


def main():
    csv_to_use = INPUT_CSV if os.path.exists(INPUT_CSV) else "output/reviews_df.csv"
    print(f"Loading {csv_to_use}...")
    df = pd.read_csv(csv_to_use)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Load existing results so we can skip already-completed hotels
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = [k for k, v in results.items() if v.get("pros")]
        print(f"Resuming: {len(already_done)} hotels already done, skipping them")
    else:
        results = {}

    hotel_ids = df["hotel_id"].unique()
    print(f"Processing {len(hotel_ids)} hotels total...")

    for i, hotel_id in enumerate(hotel_ids, start=1):
        group = df[df["hotel_id"] == hotel_id]
        hotel_name = group["hotel_name"].iloc[0]
        
        # Skip if already processed successfully
        if results.get(str(hotel_id), {}).get("pros"):
            print(f"[{i}/{len(hotel_ids)}] Skipping {hotel_name} (already done)")
            continue

        positive_reviews = group[group["rating"] >= 4]["text"].dropna().astype(str).tolist()[:MAX_REVIEWS_PER_GROUP]
        negative_reviews = group[group["rating"] <= 2]["text"].dropna().astype(str).tolist()[:MAX_REVIEWS_PER_GROUP]

        print(f"[{i}/{len(hotel_ids)}] {hotel_name} "
              f"({len(positive_reviews)} pos, {len(negative_reviews)} neg reviews)")

        if not positive_reviews and not negative_reviews:
            results[str(hotel_id)] = {"pros": [], "cons": []}
            continue

        prompt = build_prompt(hotel_name, positive_reviews, negative_reviews)
        result = call_llm(prompt)

        results[str(hotel_id)] = {
            "pros": result.get("pros", []),
            "cons": result.get("cons", []),
        }

        # Save progress periodically so a crash doesn't lose everything
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  ...progress saved ({i}/{len(hotel_ids)})")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
