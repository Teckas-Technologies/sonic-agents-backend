from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY

# ✅ Initialize OpenAI Model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")