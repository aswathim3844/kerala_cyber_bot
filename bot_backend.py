import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI #, GoogleGenerativeAIEmbeddings
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.prompts import ChatPromptTemplate
# import joblib
# from faiss import FAISS

# Load environment variables
load_dotenv()

# Load the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# System prompt for the bot
SYSTEM_PROMPT = """
You are a helpful and empathetic AI assistant designed to answer questions based on the provided context.

IMPORTANT RULES:
1. You are NOT a lawyer, financial advisor, or medical professional. Do not provide legal, financial, or medical advice.
2. You MUST base your answers only on the context provided. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Always be polite and empathetic in your responses.
4. Do not make up information or speculate beyond what is in the context.

Your response should be concise, accurate, and helpful.
"""

DISCLAIMER_TEXT = "Note: This information is provided for educational purposes only and should not be considered as professional advice."

def get_bot_response(user_question: str) -> str:
    """
    Dummy function for the frontend to connect to.
    This will be replaced with the actual implementation in Day 2.
    """
    print(f"Received query: {user_question}")
    return f"DUMMY RESPONSE for: '{user_question}'. {DISCLAIMER_TEXT}"

# Test the function
if __name__ == "__main__":
    test_question = "What is the return policy?"
    response = get_bot_response(test_question)
    print(response)