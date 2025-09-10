"""Service for prompt executor functionality (migrated from prompt-executor microservice)"""
import logging
import base64
import pickle
from typing import Tuple, List, Optional
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.prompt_executor_models import PromptTypes, GenAIModelPromptTypeMapping, PromptTemplate

logger = logging.getLogger(__name__)

# Constants from prompt-executor
DEFAULT_LLM_MODEL = "openai_gpt-3.5-turbo-1106"


class PromptExecutorService:
    """Service for handling prompt executor functionality"""
    
    @staticmethod
    def get_cvs_and_provider_model(prompt_type: str) -> Tuple[List[str], str]:
        """
        Get context variables and provider model for a prompt type
        Migrated from prompt-executor CVS endpoint
        
        Args:
            prompt_type: The type of prompt
            
        Returns:
            Tuple of (cvs, mapped_model)
        """
        db: Session = next(get_db())
        try:
            # Try to get the prompt type from database
            prompt_type_obj = db.query(PromptTypes).filter(PromptTypes.name == prompt_type).first()
            
            if not prompt_type_obj:
                # Graceful fallback when prompt type doesn't exist in database
                logger.warning(f"Prompt type '{prompt_type}' not found in database, using defaults")
                return [], DEFAULT_LLM_MODEL
            
            # Get mapped model
            mapped_model_obj = db.query(GenAIModelPromptTypeMapping).filter(
                GenAIModelPromptTypeMapping.prompt_type_id == prompt_type
            ).first()
            
            if mapped_model_obj:
                mapped_model = mapped_model_obj.genai_model
            else:
                mapped_model = DEFAULT_LLM_MODEL
                logger.warning(f"No mapped model found for prompt_type '{prompt_type}', using default: {DEFAULT_LLM_MODEL}")
            
            # Get prompt template and context variables
            prompt_template = db.query(PromptTemplate).filter(
                PromptTemplate.prompt_type_id == prompt_type
            ).first()
            
            if prompt_template and prompt_template.context_variables:
                cvs = prompt_template.context_variables.split("|")
            else:
                cvs = []
                logger.warning(f"No context variables found for prompt_type '{prompt_type}'")
            
            logger.info(f"Retrieved CVS for prompt_type '{prompt_type}': {cvs}, model: {mapped_model}")
            return cvs, mapped_model
            
        except Exception as e:
            logger.error(f"Error getting CVS and provider model for {prompt_type}: {e}")
            # Graceful fallback
            return [], DEFAULT_LLM_MODEL
        finally:
            db.close()
    
    @staticmethod
    def get_cvs_with_template(prompt_type: str) -> Tuple[List[str], str, Optional[str]]:
        """
        Get context variables, provider model, and encoded template
        Extended version of CVS endpoint that also returns the template
        
        Args:
            prompt_type: The type of prompt
            
        Returns:
            Tuple of (cvs, mapped_model, encoded_template)
        """
        db: Session = next(get_db())
        try:
            # Try to get the prompt type from database
            prompt_type_obj = db.query(PromptTypes).filter(PromptTypes.name == prompt_type).first()
            
            if not prompt_type_obj:
                # Graceful fallback when prompt type doesn't exist in database
                logger.warning(f"Prompt type '{prompt_type}' not found in database, using defaults")
                return [], DEFAULT_LLM_MODEL, None
            
            # Get mapped model
            mapped_model_obj = db.query(GenAIModelPromptTypeMapping).filter(
                GenAIModelPromptTypeMapping.prompt_type_id == prompt_type
            ).first()
            
            if mapped_model_obj:
                mapped_model = mapped_model_obj.genai_model
            else:
                mapped_model = DEFAULT_LLM_MODEL
            
            # Get prompt template
            prompt_template = db.query(PromptTemplate).filter(
                PromptTemplate.prompt_type_id == prompt_type
            ).first()
            
            if prompt_template:
                # Get context variables
                if prompt_template.context_variables:
                    cvs = prompt_template.context_variables.split("|")
                else:
                    cvs = []
                
                # Encode template like in original Django code
                pickled_response = pickle.dumps(prompt_template)
                encoded_response = base64.b64encode(pickled_response).decode('utf-8')
                
                return cvs, mapped_model, encoded_response
            else:
                logger.warning(f"No template found for prompt_type '{prompt_type}'")
                return [], mapped_model, None
                
        except Exception as e:
            logger.error(f"Error getting CVS with template for {prompt_type}: {e}")
            # Graceful fallback
            return [], DEFAULT_LLM_MODEL, None
        finally:
            db.close()
    
    @staticmethod
    def get_prompt_type(prompt_type: str) -> Optional[PromptTypes]:
        """
        Get prompt type object by name
        
        Args:
            prompt_type: Name of the prompt type
            
        Returns:
            PromptTypes object or None if not found
        """
        db: Session = next(get_db())
        try:
            return db.query(PromptTypes).filter(PromptTypes.name == prompt_type).first()
        except Exception as e:
            logger.error(f"Error getting prompt type {prompt_type}: {e}")
            return None
        finally:
            db.close()
    
    @staticmethod
    def list_prompt_types() -> List[str]:
        """
        Get list of all available prompt types
        
        Returns:
            List of prompt type names
        """
        db: Session = next(get_db())
        try:
            prompt_types = db.query(PromptTypes.name).all()
            return [pt.name for pt in prompt_types]
        except Exception as e:
            logger.error(f"Error listing prompt types: {e}")
            return []
        finally:
            db.close()