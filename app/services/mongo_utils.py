"""MongoDB utility functions"""
import logging
from typing import Optional
from bson import ObjectId
from app.core.mongodb import get_database

logger = logging.getLogger(__name__)


async def get_project_id(document_id: str) -> Optional[str]:
    """
    Get project ID from document ID
    
    Args:
        document_id: Document identifier
        
    Returns:
        Project ID if found, None otherwise
    """
    try:
        db = get_database()
        if db is None:
            logger.warning("Database not available, returning mock project_id")
            return "mock_project_id"
        
        # Query the documents collection for the project_id
        collection = db.documents
        document = await collection.find_one({"_id": ObjectId(document_id)})
        
        if document and "project_id" in document:
            return document["project_id"]
        
        logger.warning(f"Project ID not found for document {document_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting project_id for document {document_id}: {e}")
        return None


async def get_script_text(document_id: str, script_id: str) -> Optional[str]:
    """
    Get script text from document
    
    Args:
        document_id: Document identifier
        script_id: Script identifier
        
    Returns:
        Script text if found, None otherwise
    """
    try:
        db = get_database()
        if db is None:
            return None
        
        collection = db.scripts
        script = await collection.find_one({
            "document_id": document_id,
            "script_id": script_id
        })
        
        if script and "text" in script:
            return script["text"]
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting script text: {e}")
        return None