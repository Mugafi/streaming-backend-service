"""GenAI service for LLM interactions"""
import json
import logging
import openai
from typing import AsyncGenerator, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class GenAIService:
    """Service for interacting with LLM providers"""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.client = None
        
        if provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            raise ValueError(f"Provider {provider} not supported")
    
    def get_model_version(self) -> str:
        """Get the full model version string"""
        return f"{self.provider}_{self.model}"
    
    async def llm_stream_async(self, messages: list, json_schema: Optional[str] = None, 
                              is_json_output: bool = False, function_call: Optional[dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async streaming from LLM provider
        Yields chunks compatible with the existing field processing logic
        """
        try:
            if self.provider == "openai":
                # Prepare request parameters
                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True
                }
                
                if is_json_output:
                    if json_schema:
                        try:
                            schema_obj = json.loads(json_schema)
                            request_params["response_format"] = schema_obj
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON schema: {e}")
                            request_params["response_format"] = {"type": "json_object"}
                    else:
                        request_params["response_format"] = {"type": "json_object"}
                
                if function_call:
                    request_params["functions"] = [function_call]
                    request_params["function_call"] = "auto"
                
                # Stream from OpenAI
                stream = await self.client.chat.completions.create(**request_params)
                
                accumulated_content = ""
                usage_info = None
                
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        
                        if choice.delta and choice.delta.content:
                            content = choice.delta.content
                            accumulated_content += content
                            
                            yield {
                                'content': content,
                                'accumulated': accumulated_content
                            }
                        
                        if choice.finish_reason == "stop":
                            # Final output
                            final_output = accumulated_content
                            
                            # Extract usage if available
                            if hasattr(chunk, 'usage') and chunk.usage:
                                usage_info = {
                                    'prompt_tokens': chunk.usage.prompt_tokens,
                                    'completion_tokens': chunk.usage.completion_tokens,
                                    'total_tokens': chunk.usage.total_tokens
                                }
                            
                            yield {
                                'final': final_output,
                                'usage': usage_info
                            }
                            break
                            
        except Exception as e:
            logger.error(f"LLM streaming error: {str(e)}")
            yield {
                'error': str(e)
            }
    
    def extract_json(self, output: str) -> Any:
        """Extract JSON from LLM output"""
        try:
            # Try to parse the entire output as JSON
            return json.loads(output)
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the text
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            # If no valid JSON found, return the raw text
            return output