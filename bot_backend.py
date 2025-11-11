import os
import joblib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
intent_classifier = joblib.load("intent_classifier.pkl")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

SYSTEM_PROMPT = """
You are Cy-Bot, a calm and empathetic Kerala Cyber Law Assistant.
...
"""
def predict_intent(query: str) -> str:
    try:
        return intent_classifier.predict([query])[0]
    except Exception as e:
        return f"IntentError: {e}"
def get_bot_response(user_question: str) -> str:
    try:
        # 1️⃣ Classify
        intent = predict_intent(user_question)

        # 2️⃣ Retrieve context (fixed)
        docs = retriever.invoke(user_question)
        context = "\n\n".join([d.page_content for d in docs]) or "No relevant context found."

        # 3️⃣ Construct prompt
        prompt = f"{SYSTEM_PROMPT}\n\nIntent: {intent}\n\nContext:\n{context}\n\nUser Question:\n{user_question}"

        # 4️⃣ Generate response
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"[Backend Error] {str(e)}"
if __name__ == "__main__":
    print("✅ Cy-Bot backend loaded successfully — type a question below.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Cy-Bot:", get_bot_response(q))
