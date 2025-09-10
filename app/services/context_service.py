"""Context service for retrieving document context"""
import logging
from typing import Dict, Any, List
from bson import ObjectId
from app.core.mongodb import get_database

logger = logging.getLogger(__name__)


class ContextService:
    """Service for handling document context retrieval"""
    
    def __init__(self):
        self.db = get_database()
    
    async def get_context(self, document_id: str, cvs: List[str], variables: Dict[str, Any], prompt_type: str) -> str:
        """
        Get context for a document based on context variables
        
        Args:
            document_id: ID of the document
            cvs: Context variables list
            variables: Placeholder variables
            prompt_type: Type of prompt being executed
            
        Returns:
            Context text string
        """
        try:
            # This is a simplified version - implement based on your MongoDB schema
            if self.db is None:
                logger.warning("Database not available, returning mock context")
                return "Mock context for development"
            
            # Example implementation - adjust based on your document structure
            collection = self.db.documents
            document = await collection.find_one({"_id": ObjectId(document_id)})
            
            if not document:
                logger.warning(f"Document {document_id} not found")
                return ""
            
            # Build context based on CVS and variables
            context_parts = []
            
            # Add document content if available
            if "content" in document:
                context_parts.append(document["content"])
            
            # Add variable-specific context
            if variables:
                for key, value in variables.items():
                    if isinstance(value, str) and value.strip():
                        context_parts.append(f"{key}: {value}")
            
            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved context for document {document_id}, length: {len(context)}")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context for document {document_id}: {e}")
            return ""