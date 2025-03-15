
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import uuid
from app.utils.openai_utils import llm

router = APIRouter()

# ✅ Define Pydantic Model for Request Validation
class ChatRequest(BaseModel):
    abi: str = Field(..., description="The ABI of the contract")

# Initialize LangGraph AI pipeline // Return the provided ABI with an added 'description' field for each function, containing a concise one-line explanation of what the function does.
def stream_function_descriptions(abi_json): 
    try:
        system_message = SystemMessage(content="Generate concise one-line descriptions for each function in the given ABI.")
        human_message = HumanMessage(content=abi_json)
        
        def generate():
            for chunk in llm.stream([system_message, human_message]):
                print("Streaming chunk:", chunk.content)
                if hasattr(chunk, 'content'):
                        yield f"{chunk.content}"
        
        return generate()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ABI format: {str(e)}")

@router.post("/mapping")
def chat_with_agent(request: ChatRequest):
    """
    Handles ABI & creates one-line descriptions for all functions in the ABI.
    """
    return StreamingResponse(stream_function_descriptions(request.abi), media_type="text/event-stream")
