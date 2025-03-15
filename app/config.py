import os
from dotenv import load_dotenv

# ✅ Load .env variables
load_dotenv()

# ✅ Get API Keys
INFURA_API_KEY = os.getenv("INFURA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# ✅ Ensure API Keys are Set
if not INFURA_API_KEY:
    raise ValueError("⚠️ INFURA_API_KEY is missing! Set it in the .env file.")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY is missing! Set it in the .env file.")
if not MONGODB_URI:
    raise ValueError("⚠️ MONGODB_URI is missing! Set it in the .env file.")