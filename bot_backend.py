import os
import json
import joblib
from dotenv import load_dotenv

# ✅ Embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

# ✅ Google Gemini
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# ✅ INITIAL SETUP
# -------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load intent classifier
intent_classifier = joblib.load("intent_classifier.pkl")

# Load SentenceTransformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + documents
index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# ✅ Use correct working Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# -------------------------------
# ✅ HELPER FUNCTIONS
# -------------------------------

def parse_hyperlocal(text: str) -> dict:
    """Convert retrieved FAISS text into dictionary if possible."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except:
        return {}

def pretty_hyperlocal(o: dict) -> str:
    """Convert JSON fields → clean bullet points."""
    if not o:
        return "No relevant local information found."

    lines = ["📍 **Kerala Cyber Cell (Best Match)**"]

    fields = [
        ("District", o.get("district")),
        ("City", o.get("city")),
        ("Station", o.get("station_name")),
        ("Phone", o.get("phone_number")),
        ("Address", o.get("address")),
        ("Website", o.get("website"))
    ]

    for label, value in fields:
        if value:
            lines.append(f"- **{label}:** {value}")

    return "\n".join(lines)

def best_match(query: str, top_k: int = 3) -> str:
    """Return best matching FAISS document."""
    q = embedder.encode([query])
    scores, ids = index.search(q, top_k)

    for idx in ids[0]:
        if idx != -1 and str(idx) in documents:
            return documents[str(idx)]

    return ""

# -------------------------------
# ✅ MAIN BOT FUNCTION
# -------------------------------

def get_bot_response(user_question: str) -> str:
    DISCLAIMER = (
        "DISCLAIMER: I am not a lawyer. This is not legal advice. "
        "For urgent help, call 1930 or visit cybercrime.gov.in."
    )

    # 1) Intent prediction
    try:
        intent = intent_classifier.predict([user_question])[0]
    except:
        intent = "unknown"

    # 2) Retrieve one best hyperlocal chunk
    raw_doc = best_match(user_question)
    parsed = parse_hyperlocal(raw_doc)
    clean_context = pretty_hyperlocal(parsed)

    # 3) LLM prompt
    prompt = f"""
You are Cy-Bot, a calm Kerala Cyber Law assistant.

Rules:
- Never give legal advice.
- Never output raw JSON.
- Keep answers short & factual.
- Use bullet points.
- If the context already contains the answer (e.g., phone number), use it directly.
- Always end with this line: {DISCLAIMER}

Context:
{clean_context}

User question:
{user_question}

Answer:
"""

    # 4) Try Gemini
    try:
        response = llm.invoke(prompt)
        output = getattr(response, "content", "")
    except:
        output = ""

    # 5) Fallback → clean hyperlocal info
    if not output.strip():
        output = clean_context

    # 6) Ensure disclaimer exists
    if "DISCLAIMER" not in output:
        output += f"\n\n{DISCLAIMER}"

    return output
