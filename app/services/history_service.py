"""History service for ASK_VED conversation history management"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from app.core.mongodb import get_database

logger = logging.getLogger(__name__)


async def fetch_ask_ved_history_async(tag: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch ASK VED conversation history by tag
    
    Args:
        tag: Conversation tag identifier
        
    Returns:
        List of messages if found, None if not found
    """
    try:
        db = get_database()
        if db is None:
            logger.warning("Database not available for history fetch")
            return None
        
        # Query the document_ask_ved collection for conversation history
        collection = db.document_ask_ved
        result = await collection.find_one({"tag": tag})
        
        if result is None:
            logger.info(f"No conversation history found for tag: {tag}")
            return None
        
        conv_history = result.get("messages", None)
        logger.info(f"Retrieved conversation history for tag {tag}: {len(conv_history) if conv_history else 0} messages")
        return conv_history
        
    except Exception as e:
        logger.error(f"Error fetching ASK VED history for tag {tag}: {e}")
        return None


async def update_ask_ved_history_async(tag: str, messages: List[Dict[str, Any]], project_id: str, document_id: str):
    """
    Update ASK VED conversation history
    
    Args:
        tag: Conversation tag identifier
        messages: List of conversation messages
        project_id: Project identifier
        document_id: Document identifier
    """
    try:
        db = get_database()
        if db is None:
            logger.warning("Database not available for history update")
            return
        
        # Get project_id from document if not provided
        if not project_id:
            documents_collection = db.documents
            document = await documents_collection.find_one({"_id": ObjectId(document_id)})
            if document:
                project_id = document.get("project_id")
            else:
                logger.error(f"Could not find document {document_id} to get project_id")
                return
        
        # Update or insert conversation history
        collection = db.document_ask_ved
        current_time = datetime.utcnow()
        
        filter_query = {"tag": tag}
        update_query = {
            "$set": {
                "messages": messages,
                "updated_at": current_time
            },
            "$setOnInsert": {
                "_id": ObjectId(),
                "project_id": ObjectId(project_id),
                "created_at": current_time
            }
        }
        
        result = await collection.update_one(
            filter_query,
            update_query,
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"Created new conversation history for tag {tag}")
        else:
            logger.info(f"Updated conversation history for tag {tag}")
            
    except Exception as e:
        logger.error(f"Error updating ASK VED history for tag {tag}: {e}")