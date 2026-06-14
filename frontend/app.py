"""
Streamlit Frontend for Hotel & Villa Review Analyzer & Recommendation System.

Pages:
  1. Select a city  -> dropdown
  2. Select a hotel -> list of hotels in that city
  3. Hotel detail   -> sentiment score, pros, cons, chat box
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Hotel & Villa Analyzer",
    page_icon="🏨",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.pro-card {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 8px;
    color: #155724;
    font-size: 15px;
}
.con-card {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 8px;
    color: #721c24;
    font-size: 15px;
}
.chat-bubble-user {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 15px 15px 0px 15px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
}
.chat-bubble-ai {
    background-color: #f1f1f1;
    color: #333;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0px;
    margin: 6px 0;
    max-width: 80%;
}
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────
def fetch_cities():
    try:
        res = requests.get(f"{API_BASE}/cities", timeout=10)
        return res.json() if res.status_code == 200 else []
    except Exception:
        st.error(
            "Cannot connect to backend. Make sure the FastAPI server is running.")
        return []


def fetch_hotels(city):
    try:
        res = requests.get(f"{API_BASE}/cities/{city}/hotels", timeout=10)
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []


def fetch_summary(hotel_id):
    try:
        res = requests.get(f"{API_BASE}/hotels/{hotel_id}/summary", timeout=10)
        return res.json() if res.status_code == 200 else {}
    except Exception:
        return {}


def ask_question(hotel_id, question):
    try:
        res = requests.post(
            f"{API_BASE}/hotels/{hotel_id}/chat",
            json={"question": question},
            timeout=30,
        )
        return res.json().get("answer", "No answer returned.") if res.status_code == 200 else "Error getting answer."
    except Exception as e:
        return f"Error: {e}"


# ── Session state init ─────────────────────────────────────────────────────────
if "selected_hotel" not in st.session_state:
    st.session_state.selected_hotel = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏨 Sri Lanka Hotel & Villa Review Analyzer")
st.markdown(
    "AI-powered sentiment analysis and Q&A chatbot based on real TripAdvisor reviews.")
st.divider()

# ── Page 1: City selection ─────────────────────────────────────────────────────
cities = fetch_cities()

if not cities:
    st.warning("No cities loaded. Is the FastAPI backend running on port 8000?")
    st.stop()

selected_city = st.selectbox(
    "📍 Select a City",
    options=["-- Select a city --"] + sorted(cities),
)

if selected_city == "-- Select a city --":
    st.info("Please select a city to see available hotels.")
    st.stop()

# ── Page 2: Hotel list ─────────────────────────────────────────────────────────
st.subheader(f"🏩 Hotels & Villas in {selected_city}")

hotels = fetch_hotels(selected_city)

if not hotels:
    st.warning(f"No hotels found for {selected_city}.")
    st.stop()

hotel_options = {f"{h['name']} ⭐ {h['rating']}": h["hotel_id"] for h in hotels}
selected_hotel_label = st.selectbox(
    "Select a Hotel / Villa", options=["-- Select a hotel --"] + list(hotel_options.keys()))

if selected_hotel_label == "-- Select a hotel --":
    st.info("Please select a hotel to view its review analysis.")
    st.stop()

hotel_id = hotel_options[selected_hotel_label]

# Reset chat when hotel changes
if st.session_state.selected_hotel != hotel_id:
    st.session_state.selected_hotel = hotel_id
    st.session_state.chat_history = []

# ── Page 3: Hotel detail ───────────────────────────────────────────────────────
st.divider()
summary = fetch_summary(hotel_id)

if not summary:
    st.error("Could not load hotel summary.")
    st.stop()

sentiment = summary.get("sentiment", {})
pros = summary.get("pros", [])
cons = summary.get("cons", [])

hotel_name = sentiment.get("hotel_name", selected_hotel_label)
st.header(f"📊 {hotel_name}")

# ── Sentiment metrics ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("⭐ Avg Star Rating",
              f"{sentiment.get('avg_star_rating', 'N/A')} / 5")
with col2:
    st.metric("😊 Positive Reviews", f"{sentiment.get('positive_pct', 0)}%")
with col3:
    st.metric("😞 Negative Reviews", f"{sentiment.get('negative_pct', 0)}%")
with col4:
    st.metric("📝 Total Reviews", sentiment.get("total_reviews_analyzed", 0))

# Sentiment progress bar
positive_pct = sentiment.get("positive_pct", 0)
st.markdown(f"**Overall Sentiment Score: {positive_pct}% Positive**")
st.progress(int(positive_pct) / 100)

st.divider()

# ── Pros and Cons ──────────────────────────────────────────────────────────────
col_pros, col_cons = st.columns(2)

with col_pros:
    st.subheader("✅ What Guests Liked")
    if pros:
        for pro in pros:
            st.markdown(
                f'<div class="pro-card">👍 {pro}</div>', unsafe_allow_html=True)
    else:
        st.info("No pros data available for this hotel.")

with col_cons:
    st.subheader("❌ What Guests Disliked")
    if cons:
        for con in cons:
            st.markdown(
                f'<div class="con-card">👎 {con}</div>', unsafe_allow_html=True)
    else:
        st.info("No cons data available for this hotel.")

st.divider()

# ── Chat box ───────────────────────────────────────────────────────────────────
st.subheader("💬 Ask About This Hotel")
st.markdown("Ask any question and the AI will answer using real guest reviews.")

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f'<div class="chat-bubble-user">🧑 {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="chat-bubble-ai">🤖 {message}</div>', unsafe_allow_html=True)

# Input
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Your question",
        placeholder="e.g. What do guests say about the pool? Is parking available?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask 🔍")

if submitted and user_question.strip():
    st.session_state.chat_history.append(("user", user_question))
    with st.spinner("Analyzing reviews..."):
        answer = ask_question(hotel_id, user_question)
    st.session_state.chat_history.append(("ai", answer))
    st.rerun()
