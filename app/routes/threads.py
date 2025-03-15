from fastapi import APIRouter
from app.utils.mongodb_utils import fetch_user_history, delete_messages_by_thread_id
import uuid

router = APIRouter()

# Fetch all threads by user ID
@router.get("/history/{user_id}/{thread_id}")
def get_history(user_id: str, thread_id: str):  # Add type hint for clarity
    """Retrieves all thread IDs for a given user."""
    try:
        threads = fetch_user_history(user_id, thread_id)
        return {"status": 200, "threads": threads, "success": True}
    except Exception as e:
        return {"status": 500, "message": str(e), "success": False}
    
@router.delete("/history/{user_id}/{thread_id}")
def get_history(user_id: str, thread_id: str):  # Add type hint for clarity
    """Delete all messages for a given user."""
    try:
        threads = delete_messages_by_thread_id(user_id, thread_id)
        return {"status": 200, "threads": threads, "success": True}
    except Exception as e:
        return {"status": 500, "message": str(e), "success": False}