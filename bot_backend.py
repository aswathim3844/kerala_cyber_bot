import os
import joblib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # ✅ new import

# Load environment variables
load_dotenv()

# Load your ML classifier
intent_classifier = joblib.load("intent_classifier.pkl")

# Use local Hugging Face embeddings (no API limit)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3) # ⚡ Fast & free

# System Prompt
SYSTEM_PROMPT = """
You are Cy-Bot, a helpful and polite AI assistant that provides information about Kerala's cyber laws.

Rules:
1. You are NOT a lawyer — never give personal legal advice.
2. Use only the provided context to answer questions.
3. Write in simple, clear English.
4. If giving penalties or procedures, use bullet points.
5. End every answer with:
   "⚠️ This is not legal advice. For urgent help, call 1930 or visit cybercrime.gov.in."
"""

# Predict intent
def predict_intent(query: str) -> str:
    try:
        return intent_classifier.predict([query])[0]
    except Exception as e:
        return f"IntentError: {e}"

# Generate bot response
def get_bot_response(user_question: str) -> str:
    try:
        # 1️⃣ Intent prediction
        intent = predict_intent(user_question)

        # 2️⃣ Retrieve relevant documents
        try:
            docs = retriever.invoke(user_question)
        except AttributeError:
            docs = retriever.get_relevant_documents(user_question)

        context = "\n\n".join([d.page_content for d in docs]) or "No relevant context found."

        # 3️⃣ Build prompt
        prompt = f"""
{SYSTEM_PROMPT}

Intent: {intent}

Context:
{context}

Question: {user_question}

Answer (structured and clear):
"""

        # 4️⃣ Generate answer using Groq
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"[Backend Error] {str(e)}"

# Local testing
if __name__ == "__main__":
    print("✅ Cy-Bot (Groq) backend loaded successfully — type a question below.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Cy-Bot:", get_bot_response(q))
