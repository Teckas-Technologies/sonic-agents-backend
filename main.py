from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import agents, chat, threads, mapping
from app.utils.agent_utils import load_agents_on_startup
from app.utils.web3_utils import read_from_contract

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URLs if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allows all headers
)


app.include_router(agents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(threads.router, prefix="/api")
app.include_router(mapping.router, prefix="/api")
load_agents_on_startup()


@app.get("/")
def root():
    return {"message": "Agentify-AI is running!"}


read_from_contract()
