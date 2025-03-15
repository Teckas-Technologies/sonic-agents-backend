from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from app.utils.agent_utils import AGENT_REGISTRY, get_relevant_tool_message, get_last_ai_message, get_last_message
from langchain_core.messages import HumanMessage, AIMessage
import uuid

router = APIRouter()


# ✅ Define Pydantic Model for Request Validation
class ChatRequest(BaseModel):
    agentName: str = Field(..., description="The name of the agent")
    userId: str = Field(..., description="User's unique ID")
    message: str = Field(..., description="User's input message")
    threadId: str = Field(None, description="Thread ID for the conversation")  # Optional


@router.post("/chat")
def chat_with_agent(request: ChatRequest, response: Response):
    """
    Handles user messages, maintains conversation threads, and routes to the appropriate AI agent.
    """
    agent_name = request.agentName
    user_id = request.userId
    user_input = request.message
    thread_id = request.threadId or str(uuid.uuid4())  # Generate a new thread ID if missing

    # ✅ Ensure agent exists
    if agent_name not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found!")

    config = {"configurable": {"thread_id": f"{user_id}-{thread_id}"}}

    # ✅ Process the message with the agent
    agent_graph = AGENT_REGISTRY[agent_name]["graph"]
    # responses = [output for output in agent_graph.stream({"messages": HumanMessage(content=user_input)}, config)]
    # print("res: ", response)
    # ✅ Extract AI or Blockchain response
    # tool_response = extract_first_tool_message(responses)
    # if tool_response:
    #     tool_response["threadId"] = thread_id  # Update with actual thread ID
    #     response.status_code = 201
    #     return tool_response

    # ai_response = extract_first_ai_response(responses)
    # ai_response["threadId"] = thread_id  # Update with actual thread ID

    response = agent_graph.invoke({"messages": HumanMessage(content=user_input)}, config)
    last_message = response.get("messages", [])[-1]
    last_message = get_last_message(last_message)
    print("Last Message Content", last_message)

    return last_message

    # responses = [output for output in agent_graph.stream({"messages": HumanMessage(content=user_input)}, config)]

    # # ✅ Extract the last AI message
    # ai_response = get_last_ai_message_stream(responses)

    # return ai_response
