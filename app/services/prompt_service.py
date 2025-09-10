"""Service for prompt-related operations"""
import logging
from typing import Tuple, List
from app.services.prompt_executor_service import PromptExecutorService

logger = logging.getLogger(__name__)


class PromptService:
    """Service for prompt management operations"""
    
    async def get_cvs_and_provider_model(self, prompt_type: str) -> Tuple[List[str], str]:
        """
        Get context variables and provider model for a prompt type
        Async wrapper around the PromptExecutorService method
        
        Args:
            prompt_type: The type of prompt
            
        Returns:
            Tuple of (cvs, mapped_model)
        """
        try:
            return PromptExecutorService.get_cvs_and_provider_model(prompt_type)
        except Exception as e:
            logger.error(f"Error in get_cvs_and_provider_model: {e}")
            return [], "openai_gpt-4o-mini"