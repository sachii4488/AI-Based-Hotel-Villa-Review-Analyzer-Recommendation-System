# 🏨 AI-Based Hotel & Villa Review Analyzer & Recommendation System

An AI-powered web application that analyzes real hotel and villa reviews from Sri Lanka using Natural Language Processing (NLP), Transformer Models, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs).

The system helps travelers make informed accommodation decisions by providing sentiment analysis, AI-generated pros & cons summaries, and an intelligent chatbot that answers questions based on actual guest experiences.

---

## 🚀 Project Overview

This project processes and analyzes TripAdvisor hotel and villa reviews collected from Sri Lanka. Users can browse hotels by city, view sentiment insights from guest reviews, explore AI-generated summaries of strengths and weaknesses, and interact with a chatbot grounded in real review data.

The chatbot uses Retrieval-Augmented Generation (RAG) to retrieve relevant reviews and generate trustworthy responses based only on available review content.

---

## ✨ Features

### 📍 City & Hotel Selection

* Browse **108 hotels and villas**
* Explore accommodations across **22 Sri Lankan cities**

### 😊 Sentiment Analysis

* Analyze individual reviews using DistilBERT
* Classify reviews as:

  * Positive
  * Negative
* Display confidence scores for each prediction

### 👍👎 Pros & Cons Generation

* Automatically identify recurring positive and negative themes
* Generate concise summaries from multiple guest reviews
* Present results as easy-to-read bullet points

### 🤖 RAG-Based Chatbot

* Ask questions about any hotel
* Retrieve relevant reviews using semantic search
* Generate answers grounded only in actual guest feedback
* Minimize hallucinations through retrieval-based prompting

### 🌐 Interactive Web Interface

* FastAPI REST backend
* Streamlit frontend
* Real-time AI-powered analysis and recommendations

---

## 🧠 AI Techniques Implemented

### 1. Natural Language Processing (NLP)

#### Text Pre-processing

Implemented in `prepare_data.py`

* Cleaning raw review text
* Removing null values
* Deduplication
* Hotel name normalization
* City name standardization

#### Text Post-processing

Implemented in `sentiment_analysis.py`

* Tokenization
* Truncation to 512 tokens
* Sentiment label extraction

#### Review Filtering & Aggregation

Implemented in `pros_cons.py`

* Rating-based review filtering
* Review grouping
* Context preparation for LLM summarization

---

### 2. Transformer-Based Models & LLMs

#### DistilBERT

Model:
`distilbert-base-uncased-finetuned-sst-2-english`

Purpose:

* Sentiment classification
* Transfer learning for hotel review analysis

#### Sentence Transformer

Model:
`all-MiniLM-L6-v2`

Purpose:

* Semantic embeddings
* Similarity search
* Review retrieval for RAG

#### Llama 3.3 70B

Accessed through:

* Groq API

Purpose:

* Pros & cons generation
* Conversational question answering
* Context-aware response generation

---

### 3. Prompt Engineering

#### Structured Output Prompting

Implemented in `pros_cons.py`

* JSON-formatted responses
* Fixed output structure
* Controlled bullet counts
* Word-length constraints

#### Retrieval-Augmented Prompting

Implemented in `chatbot.py`

* Inject retrieved reviews into prompts
* Ground responses using real review evidence
* Restrict answers to retrieved context

#### Hallucination Prevention

Includes fallback instructions such as:

> "If there is not enough information in the reviews, say so."

---

## 🛠️ Technology Stack

| Layer                | Technology                          |
| -------------------- | ----------------------------------- |
| Programming Language | Python 3.11+                        |
| Sentiment Model      | DistilBERT                          |
| Embedding Model      | all-MiniLM-L6-v2                    |
| Vector Database      | ChromaDB                            |
| Large Language Model | Llama 3.3 70B (Groq)                |
| Backend API          | FastAPI                             |
| Frontend             | Streamlit                           |
| Server               | Uvicorn                             |
| Dataset              | TripAdvisor Sri Lanka Hotel Reviews |

---

## 📊 Dataset

**Dataset Name:** TripAdvisor Sri Lanka Hotel Reviews

Dataset Source:

https://www.kaggle.com/datasets/nethumdperera/travel-destinations-reviews-in-sir-lanka

### Dataset Statistics

* **3,558 reviews**
* **108 hotels & villas**
* **22 Sri Lankan cities**

### Fields Used

| Field          | Description               |
| -------------- | ------------------------- |
| Review Text    | Customer review content   |
| Star Rating    | Numerical rating          |
| Trip Type      | Type of travel            |
| Published Date | Review publication date   |
| Hotel Name     | Accommodation name        |
| Location       | City/location information |
| Coordinates    | Geographic coordinates    |

---


## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/hotel-review-analyzer.git

cd hotel-review-analyzer
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Running the Application

### Start FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend URL:

```text
http://localhost:8000
```

### Start Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## 🎯 Future Enhancements

* Hotel recommendation ranking system
* Aspect-based sentiment analysis
* Multi-language review support
* Review trend visualization
* Personalized travel recommendations
* Voice-enabled travel assistant

---




