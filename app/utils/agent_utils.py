import json
from typing import Dict, TypedDict, Annotated
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, AnyMessage
from langchain.tools import Tool
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from app.utils.openai_utils import llm
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from app.config import MONGODB_URI
import requests

AGENT_REGISTRY = {}

ALLOWED_TOKENS = {
    "WSOL": "So11111111111111111111111111111111111111112",
    "USDT": "qPzdrTCvxK3bxoh2YoTZtDcGVgRUwm37aQcC3abFgBy",
    "USDC": "HbDgpvHVxeNSRCGEUFvapCYmtYfqxexWcCbxtYecruy8",
    "SONIC": "mrujEYaN1oyQXDHeYNxBYpxWKVkQ2XsGxfznpifu4aL",
    "sonicSOL": "CCaj4n3kbuqsGvx4KxiXBfoQPtAgww6fwinHTAPqV5dS",
    "sSOL": "DYzxL1BWKytFiEnP7XKeRLvgheuQttHW643srPG6rNRn",
    "lrtsSOL": "7JPHd4DQMwMnFSrKJQZzqabcrWfuRvsuWsxwuGbbmFfR",
}

ALLOWED_SWAPS = {
    ("WSOL", "SONIC"): "DgMweMfMbmPFChTuAvTf4nriQDWpf9XX3g66kod9nsR4",
    ("WSOL", "sonicSOL"): "PVefkK6H5CL4qyBKmJaqFKYGM2bhzkpfwRhnMEUgXy4",
    ("USDT", "USDC"): "Q3xCT8sxU9VVYfHSmfQDahwGe9e43Q6hfEpLLi15yLo",
    ("WSOL", "lrtsSOL"): "BeqkciWYCaAtn2FRaXrhfkWnWSgjnBB96ovZ4toJCTKW",
    ("WSOL", "USDT"): "28UY6pGdJxvhhg1L45nrDe8j6RPUBuneL51RJ18NBcoz",
    ("SONIC", "USDT"): "3htP3trfRffLQSYf3ixJD3kx7kkKwEyPYJceCodt9UBz",
}

TOKEN_DESCRIPTION = "\n".join([f"- {name} ," for name, address in ALLOWED_TOKENS.items()])
SWAP_DESCRIPTION = "\n".join([f"- {pair[0]} → {pair[1]}" for pair, pool_id in ALLOWED_SWAPS.items()])


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    last_agent: str


def create_sonic_agent():
    def prepare_swap_transaction(*args) -> Dict:
        """
        Validates swap parameters and prepares a swap transaction using Sega SDK.
        """
        params = json.loads(args[0])
        required_fields = ["fromToken", "toToken", "amount"]
        for field in required_fields:
            if field not in params or not params[field]:
                return {"error": f"Missing required field: {field}"}
        # Extract parameters
        from_token = params["fromToken"]
        to_token = params["toToken"]
        amount = params["amount"]

        # Validate token selection
        if from_token not in ALLOWED_TOKENS.keys() or to_token not in ALLOWED_TOKENS.keys():
            return {
                "error": "Invalid tokens. Allowed tokens are:\n" + "\n".join(
                    [f"{name}: {address}" for name, address in ALLOWED_TOKENS.items()]
                )
            }

        swap_key = (from_token, to_token)
        print("Swap Key: ", swap_key)
        print("ALLOWED SWAPS", ALLOWED_SWAPS.keys())
        if swap_key not in ALLOWED_SWAPS.keys():
            return {
                "error": f"Invalid swap pair. Allowed swaps are:\n" + "\n".join(
                    [f"{pair[0]} → {pair[1]} (Pool: {pool_id})" for pair, pool_id in ALLOWED_SWAPS.items()]
                )
            }

        print("Swap selected.....")
        # Get correct pool ID
        pool_id = ALLOWED_SWAPS.get(swap_key)

        # Return formatted swap transaction
        return {
            "fromToken": from_token,
            "fromTokenId": ALLOWED_TOKENS.get(from_token),
            "toToken": to_token,
            "toTokenId": ALLOWED_TOKENS.get(to_token),
            "amount": amount,
            "poolId": pool_id
        }

    # ✅ Define Bridge Tool (Agent collects inputs & formats message)
    def prepare_bridge_request(*args) -> Dict:
        """Prepares a cross-chain bridge request between Sonic and Solana."""
        params = json.loads(args[0])
        print("🔹 Bridge Request Initiated:", params)

        required_fields = ["fromChain", "amount"]
        for field in required_fields:
            if field not in params or not params[field]:
                return {"error": f"Missing required field: {field}"}
        recipientAddress = ""
        if "recipientAddress" in params:
            recipientAddress = params["recipientAddress"]
        return {
            "fromChain": params["fromChain"],
            "amount": params["amount"],
            "recipientAddress": recipientAddress
        }

    # ✅ Define Bridge Tool
    bridge_tool = Tool(
        name="bridge_assets",
        func=prepare_bridge_request,
        description="Collects details for a cross-chain bridge transaction. "
                    "Ensure the following details are provided: "
                    "- `fromChain`: The chain you're bridging from. Only Solana and Sonic chains are allowed. If user "
                    "provides any different chain then tell them only Solana and Sonic or Sonic are allowed."
                    "- `amount`: The amount of tokens to bridge. "
                    "- `recipientAddress`: The recipient wallet address on the destination chain. This is optional. User "
                    "May or may not provide the recipient address"
                    "Call this tool only when all parameters are correctly provided."
    )

    swap_tool = Tool(
        name="swap_tokens",
        func=prepare_swap_transaction,
        description=f"Prepares a token swap transaction on Sega SDK.\n\n"
                    f" **Allowed Tokens:**\n{TOKEN_DESCRIPTION}\n\n"
                    f" **Allowed Swap Pairs:**\n{SWAP_DESCRIPTION}\n\n"
                    "- `fromToken` & `toToken` must be valid token names or addresses.\n"
                    "- `amount` must be in the smallest denomination (lamports).\n"
                    "- Only specific token pairs are allowed for swapping.\n"
                    "Call this tool only when all parameters are correctly provided."
    )

    bridge_agent = llm.bind_tools([bridge_tool])
    swap_agent = llm.bind_tools([swap_tool])

    def supervisor(state: AgentState):
        """Routes requests based on user intent, tracking selected agent."""
        user_input = state["messages"][-1].content
        last_agent = state.get("last_agent", None)

        if last_agent == "swap_agent" or last_agent == "bridge_agent":
            # ✅ If an agent is already selected, route to it
            return {"next": last_agent, "last_agent": last_agent}

        # ✅ Otherwise, determine intent
        prompt = f"User Request: '{user_input}'\n\nRespond with 'swap', 'bridge', or 'chat'."
        decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()

        if decision in ["swap", "bridge"]:
            return {"messages": state["messages"][-1], "next": decision + "_agent", "last_agent": decision + "_agent"}

        return {"next": "chat", "last_agent": None}

    def bridge_agent_node(state: AgentState):
        """Handles user inputs, collects missing parameters, and executes bridge when ready."""

        user_input = state["messages"][-1].content

        intent_check_prompt = (
            f"User said: '{user_input}'.\n\n"
            "Does the user still want to proceed with the token bridge, or do they want to exit? "
            "Reply with only one word: 'bridge' or 'chat'."
        )
        decision = llm.invoke([SystemMessage(content=intent_check_prompt)]).content.strip().lower()

        if decision == "chat":
            print("🔹 LLM determined user no longer wants to bridge. Routing to Chat Agent.")
            return {"next": "supervisor", "last_agent": None}

        response = bridge_agent.invoke(state["messages"])  # AI processes input and decides when to call the tool
        print("Response: ", response)
        if response.tool_calls:
            return {
                "messages": state["messages"] + [response],
                "next": "bridge_tool",
                "last_agent": "END"
            }
        return {
            "messages": state["messages"] + [response],
            "next": "END",
            "last_agent": "bridge_agent"
        }

    def swap_agent_node(state: AgentState):
        """Handles user inputs, collects missing parameters, and executes swap when ready."""
        user_input = state["messages"][-1].content

        intent_check_prompt = (
            f"User said: '{user_input}'.\n\n"
            "Does the user still want to proceed with the token swap, or do they want to exit? "
            "Reply with only one word: 'swap' or 'chat'."
        )
        decision = llm.invoke([SystemMessage(content=intent_check_prompt)]).content.strip().lower()

        if decision == "chat":
            print("🔹 LLM determined user no longer wants to swap. Routing to Chat Agent.")
            return {"next": "supervisor", "last_agent": None}

        response = swap_agent.invoke(state["messages"])  # AI processes input and decides when to call the tool
        print("Response: ", response)
        if response.tool_calls:
            return {
                "messages": state["messages"] + [response],
                "next": "swap_tool",
                "last_agent": "END"
            }
        return {
            "messages": state["messages"] + [response],
            "next": "END",
            "last_agent": "swap_agent"
        }

    def chat_agent(state: AgentState):
        """Handles general conversations."""
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response], "next": "END"}

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("swap_agent", swap_agent_node)
    workflow.add_node("swap_tool", ToolNode([swap_tool]))
    workflow.add_node("bridge_agent", bridge_agent_node)
    workflow.add_node("bridge_tool", ToolNode([bridge_tool]))
    workflow.add_node("chat", chat_agent)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", lambda state: state["next"], {
        "swap_agent": "swap_agent",
        "bridge_agent": "bridge_agent",
        "chat": "chat",
        "END": END
    })

    workflow.add_conditional_edges("swap_agent", lambda state: state["next"], {
        "swap_tool": "swap_tool",
        "supervisor": "supervisor",
        "END": END
    })

    workflow.add_conditional_edges("swap_tool", lambda state: "END", {"END": END})
    workflow.add_conditional_edges("bridge_agent", lambda state: state["next"], {
        "bridge_tool": "bridge_tool",
        "supervisor": "supervisor",
        "END": END
    })
    workflow.add_conditional_edges("bridge_tool", lambda state: "END", {"END": END})

    workflow.add_edge("chat", END)

    workflow.compile()

    # ✅ Compile Graph
    mongodb_client = MongoClient(MONGODB_URI, tlsAllowInvalidCertificates=True, tlsAllowInvalidHostnames=True)
    checkpointer = MongoDBSaver(
        mongodb_client,
        db_name="new_memory_new",
        checkpoint_ns="AGY"
    )

    # ✅ Compile Graph & Register Multi-ABI Agent
    graph = workflow.compile(checkpointer=checkpointer)
    AGENT_REGISTRY["sonicAgent"] = {"graph": graph}
    return graph


def get_last_ai_message(response_data):
    """
    Extracts the content of the last AIMessage from the response data.
    """
    messages = response_data.get("messages", [])

    # Iterate in reverse to find the last AI message
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:  # Ensure it's an AI message with content
            return message.content

    return "No valid AI response found."


def get_last_message(message):
    response_payload = {
        "ai_message": "None",
        "tool_response": "None"
    }
    if isinstance(message, AIMessage) and message.content:
        response_payload = {
            "ai_message": message.content,
            "tool_response": "None"
        }
    if isinstance(message, ToolMessage) and message.content:
        response_payload = {
            "ai_message": "None",
            "tool_response": message.content
        }
    return response_payload


def get_relevant_tool_message(response_data):
    """
    Extracts the last ToolMessage before the last AIMessage.
    If the last AIMessage doesn't have a preceding ToolMessage, find the most recent one before it.
    """
    messages = response_data.get("messages", [])

    last_ai_index = None
    last_tool_message = None

    # Iterate in reverse to locate the last AIMessage
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]

        if isinstance(message, AIMessage):
            if last_ai_index is None:
                last_ai_index = i  # Store the index of the last AIMessage
            else:
                break  # Stop when encountering an earlier AIMessage

        if isinstance(message, ToolMessage):
            last_tool_message = message.content  # Store the most recent ToolMessage

        # If we found an AIMessage and already stored a ToolMessage before it, return it
        if last_ai_index is not None and last_tool_message:
            return last_tool_message

    return "None"


def load_agents_on_startup():
    """
    Loads agents from the database into memory on startup.
    """
    create_sonic_agent()
    print("Agents Loaded Successfully")
