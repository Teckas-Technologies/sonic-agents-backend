from fastapi import APIRouter, HTTPException
from app.utils.agent_utils import AGENT_REGISTRY, load_agents_on_startup
from app.utils.mongodb_utils import update_agent_to_db

router = APIRouter()


@router.post("/create-agent")
def create_agent(agent_data: dict):
    if agent_data["agentName"] in AGENT_REGISTRY:
        raise HTTPException(status_code=400, detail="Agent already exists!")

    # create_dynamic_agent(agent_data["agentId"], agent_data["agentName"], agent_data["blockchain"], agent_data["functionMappings"], agent_data["abiDetails"])
    return {"message": f"Agent '{agent_data['agentName']}' created successfully!"}


@router.put("/update-agent/{agent_id}")
def create_agent(agent_id: str, agent_data: dict):
    update_agent_to_db(agent_id, agent_data["agentName"], agent_data["blockchain"], agent_data["abiDetails"],
                       agent_data["functionMappings"])
    return {"message": f"Agent '{agent_data['agentName']}' updated successfully!"}


@router.get("/list-agents")
def list_agents():
    """API to fetch all registered agents."""
    return {"agents": list(AGENT_REGISTRY.keys())}


@router.get("/load-agents")
def list_agents():
    load_agents_on_startup()
    """API to fetch all registered agents."""
    return {"agents": list(AGENT_REGISTRY.keys())}
