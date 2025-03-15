from pymongo import MongoClient
from app.config import MONGODB_URI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Connect to MongoDB
mongodb_client = MongoClient(MONGODB_URI)
db = mongodb_client.get_database("memory_db")  # Database name
messages_collection = db["messages"]  # Collection name
agents_collection = db["agents"]


# ✅ Store Agent in DB
def save_agent_to_db(agent_id, agent_name, blockchain, abi_details, function_mappings):
    """
    Stores agent details in MongoDB.
    """
    agent_data = {
        "agentName": agent_name,
        "blockchain": blockchain,
        "abiDetails": abi_details,
        "functionMappings": function_mappings,
        "agentId": agent_id

    }
    agents_collection.update_one({"agentName": agent_name}, {"$set": agent_data}, upsert=True)


def update_agent_to_db(agent_id: str, agent_name: str, blockchain: str, abi_details: dict, function_mappings: str):
    """
    Stores agent details in MongoDB.
    """
    agent_data = {
        "agentName": agent_name,
        "blockchain": blockchain,
        "abiDetails": abi_details,
        "functionMappings": function_mappings,
        "agentId": agent_id
    }
    agents_collection.update_one({"agentId": agent_id}, {"$set": agent_data}, upsert=True)


# ✅ Load All Agents from DB
def load_agents_from_db():
    """
    Loads all agents from MongoDB.
    """
    agents = list(agents_collection.find({}, {"_id": 0}))  # ✅ Fetch all agents, exclude MongoDB `_id`

    if not agents:
        print("⚠️ No agents found in MongoDB.")

    return agents


# ✅ Get a Specific Agent
def get_agent_from_db(agent_name):
    """
    Retrieves a specific agent from MongoDB.
    """
    return agents_collection.find_one({"agentName": agent_name}, {"_id": 0})


def fetch_user_history(user_id, thread_id):
    config = {"configurable": {"thread_id": f"{user_id}-{thread_id}"}}

    checkpointer = MongoDBSaver(
        mongodb_client, 
        db_name="new_memory",
        checkpoint_ns="AGY"
    )

    history = checkpointer.get_tuple(config)

    if history is None:
        print("🚨 ERROR: Checkpoint tuple is None!")
        return []

    print("CHECKPOINT TUPLE TYPE:", type(history))

    if hasattr(history, 'checkpoint'):
        checkpoint_data = history.checkpoint  # Access the dictionary part
    elif isinstance(history, tuple):
        print("CHECKPOINT TUPLE LENGTH:", len(history))
        checkpoint_data = history[1]  # Assuming second element is the dictionary
    else:
        checkpoint_data = history  # Use it directly if it's already a dictionary

    if checkpoint_data is None:
        print("🚨 ERROR: Checkpoint data is None!")
        return []

    print("CHECKPOINT DATA KEYS:", checkpoint_data.keys())

    messages = checkpoint_data.get('channel_values', {}).get('messages', [])

    print("EXTRACTED MESSAGES:", messages)

    parsed_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        # elif isinstance(message, ToolMessage):
        #     role = "tool"
        else:
            print("UNKNOWN MESSAGE TYPE:", type(message))
            continue

        parsed_messages.append({"role": role, "message": message.content, "message_id": message.id})

    return parsed_messages

def delete_messages_by_thread_id(user_id, thread_id):
    try:
        db = mongodb_client.get_database("new_memory")
        collection = db["checkpoints"]  # Assuming "AGY" is the collection name
        
        result = collection.delete_many({"thread_id": f"{user_id}-{thread_id}" })

        print("RES:", result)
        
        if result.deleted_count > 0:
            print(f"✅ Successfully deleted {result.deleted_count} messages for thread_id: {thread_id}")
            return True
        else:
            print("🚨 No messages found for the given thread_id!")
            return False
    except Exception as e:
        print(f"🚨 ERROR: Failed to delete messages - {e}")
        return False
