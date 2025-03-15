# 🚀  Sonic Agents

A **FastAPI-powered** agent API for blockchain transactions. This API allows developers to create blockchain agents dynamically using **smart contract ABIs** and interact with them via chat-based commands.

---

## 📂 Project Structure

```
sonic-agents-backend/
│── app/
│   │── __init__.py
│   │── main.py       # Main FastAPI entry point
│   │── config.py     # Loads environment variables from .env
│   │── routes/
│   │   │── __init__.py
│   │   │── agents.py # API for agent creation
│   │   │── chat.py   # API for chat interactions
│   │── utils/
│   │   │── __init__.py
│   │   │── web3_utils.py  # Web3 helper functions
│   │   │── openai_utils.py # OpenAI helper functions
│── .env           # Stores API keys (DO NOT COMMIT)
│── .gitignore     # Ignore venv, .env, etc.
│── requirements.txt # List of dependencies
│── README.md      # Documentation
│── start.sh       # Startup script for deployment
│── Dockerfile     # Docker configuration
```

---

## ✅ **1. Setup Virtual Environment (`venv`)

1️⃣ **Navigate to the project directory:**
```bash
cd ~/sonic-agents-backend
```

2️⃣ **Create and activate a virtual environment:**
```bash
python -m venv venv  # Create venv
source venv/bin/activate  # Activate venv (Mac/Linux)
venv\Scripts\Activate  # (Windows PowerShell)
```

3️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```

4️⃣ **Save installed packages:**
```bash
pip freeze > requirements.txt
```

---

## 🚀 **2. Running FastAPI Locally**

1️⃣ **Ensure `.env` file exists with API keys:**
```
INFURA_API_KEY=your_infura_api_key
OPENAI_API_KEY=your_openai_api_key
```

2️⃣ **Run the FastAPI server using Uvicorn:**
```bash
uvicorn app.main:app --reload
```

📌 **Open in Browser:**
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **Redoc UI:** `http://127.0.0.1:8000/redoc`

---

## 🛠 **3. Running FastAPI with Gunicorn (Production Mode)**

1️⃣ **Install Gunicorn:**
```bash
pip install gunicorn
```

2️⃣ **Run with Gunicorn & Uvicorn Workers:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

📌 This starts **4 workers** for handling requests efficiently.

---

## 🐳 **4. Running FastAPI with Docker**

1️⃣ **Create a `Dockerfile`:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2️⃣ **Build the Docker image:**
```bash
docker build -t blockchain-agent-api .
```

3️⃣ **Run the Docker container:**
```bash
docker run -p 8000:8000 blockchain-agent-api
```

---

## 🌍 **5. Deploying on a Linux Server (AWS/GCP/DO)**

1️⃣ **Ensure Python is installed** (Python 3.10+ recommended)
```bash
sudo apt update && sudo apt install python3 python3-venv -y
```

2️⃣ **Clone the repository & set up venv**
```bash
git clone https://github.com/your-repo/blockchain-agent-api.git
cd blockchain-agent-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3️⃣ **Run FastAPI with Gunicorn (daemon mode)**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app --daemon
```

4️⃣ **Enable Firewall (if required)**
```bash
sudo ufw allow 8000/tcp
```

Now access the API at `http://server-ip:8000` 🚀

---

## 🚀 **6. Deploy with Docker on a Server**

1️⃣ **Install Docker (if not installed)**
```bash
sudo apt update && sudo apt install docker.io -y
```

2️⃣ **Run FastAPI inside Docker:**
```bash
docker build -t blockchain-agent-api .
docker run -d -p 8000:8000 blockchain-agent-api
```

✅ Your API is now running inside a **Docker container** and accessible from anywhere!

---

## 🏆 **Done! Your Sonic Angent AI API is Ready!** 🎯
- Use `venv` for local development.
- Use `Gunicorn` for **high-performance production** deployment.
- Use **Docker** for **containerized deployment** on any cloud.

🚀 **Now your FastAPI project is production-ready and can be deployed anywhere!** 🚀

