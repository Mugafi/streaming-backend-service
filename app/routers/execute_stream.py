"""Router for prompt management endpoints - EXACT copy of core-backend structure"""
import json
import logging
import asyncio
import uuid
import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.prompt_models import ExecuteStreamRequest
from app.services.context_service import ContextService
from app.services.prompt_service import PromptService
from app.services.prompt_executor_service import PromptExecutorService
from app.services.genai_service import GenAIService
from app.services.mongo_utils import get_project_id
from app.services.history_service import fetch_ask_ved_history_async, update_ask_ved_history_async
from app.core.logging import get_logger
from app.core.constants import CONSTANTS

logger = get_logger(__name__)
router = APIRouter()

# Convert sync functions to async (same as core-backend)
context_service = ContextService()
prompt_service = PromptService()


@router.post("/execute-async-stream")
async def execute_stream(request_data: ExecuteStreamRequest):
    """
    FastAPI endpoint for async streaming prompt execution
    EXACT copy of core-backend async_execute_stream logic
    """
    logger.info(f"Executing async stream for document_id: {request_data.document_id}")
    
    # Extract parameters (EXACT same as core-backend)
    customer_id = request_data.customer_id
    # Note: FastAPI doesn't have request.user like Django - customer_id should come from request_data
    
    document_id = request_data.document_id
    lang = request_data.lang or ''
    
    if not document_id:
        raise HTTPException(
            status_code=400,
            detail={
                "status": -1,
                "message": "project information is required."
            }
        )
    
    prompt_type = request_data.prompt_type
    variables = request_data.placeholders
    project_id = request_data.project_id
    input_tag = request_data.tag
    
    if project_id is None:
        project_id = await get_project_id(document_id)
    
    # Handle conversation history for specific prompt types (EXACT same)
    prev_messages = None
    return_prompt = False
    if (prompt_type == CONSTANTS.ASK_VED_PROMPT or 
        prompt_type == CONSTANTS.ASK_VED_SCRIPT_PROMPT or 
        prompt_type.endswith(CONSTANTS.EXPANSION_SUFFIX) or 
        prompt_type.endswith(CONSTANTS.SUGGEST_NEXT_SUFFIX)):
        
        page_num = request_data.page_num
        if not page_num:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": -1,
                    "message": f"page_num information is required for {prompt_type}."
                }
            )
        return_prompt = True
        if input_tag:
            prev_messages = await fetch_ask_ved_history_async(input_tag)
            if prev_messages:
                page_num = 2*page_num-1
                prev_messages = prev_messages[:page_num]
    
    # Variable preprocessing (EXACT same)
    if variables and "scene" in variables and isinstance(variables["scene"], list):
        variables["scene"] = f"`json_schema: {json.dumps(variables['scene'])}`"
    
    if variables and 'complete_scene' in variables and isinstance(variables['complete_scene'], str):
        if variables['complete_scene'].strip().lower().endswith('action:'):
            variables['complete_scene'] = variables['complete_scene'].rstrip().rsplit('action:', 1)[0].rstrip()

    # EXACT same async_stream_generator function as core-backend
    async def async_stream_generator():
        """True async generator for real-time streaming - EXACT copy from core-backend"""
        
        def clean_json_value(value):
            """Clean malformed JSON values like escaped quotes, extra commas etc."""
            if not isinstance(value, str):
                logger.debug(f"🧽 clean_json_value: non-string value {type(value)}: {value}")
                return value
            
            logger.debug(f"🧽 clean_json_value: input='{value}' ({len(value)} chars)")
            
            # Remove trailing escaped quotes and commas
            cleaned = value.rstrip('\",').rstrip('",').rstrip('\\"').rstrip('"')
            
            # Unescape common JSON escape sequences
            cleaned = cleaned.replace('\\"', '"')
            cleaned = cleaned.replace('\\n', '\n')
            cleaned = cleaned.replace('\\t', '\t')
            cleaned = cleaned.replace('\\r', '\r')
            cleaned = cleaned.replace('\\\\', '\\')
            
            logger.debug(f"🧽 clean_json_value: output='{cleaned}' ({len(cleaned)} chars)")
            return cleaned
        
        class StreamingWordBuffer:
            """Buffer content and stream complete words with proper newline handling"""
            
            def __init__(self, buffer_size=7, max_word_size=50):
                self.buffer = ""
                self.buffer_size = buffer_size
                self.max_word_size = max_word_size
            
            def add_content(self, new_content):
                """Add content to buffer and return complete words ready for streaming"""
                if not isinstance(new_content, str):
                    return []
                
                # Add to buffer
                self.buffer += new_content
                
                # Clean the buffer to handle escape sequences
                cleaned_buffer = clean_json_value(self.buffer)
                
                if not cleaned_buffer:
                    return []
                    
                words_to_stream = []
                
                # Wait for minimum buffer size unless we have newlines or word is very long
                if len(cleaned_buffer) < self.buffer_size and '\n' not in cleaned_buffer:
                    # Check for very long words that exceed max_word_size
                    if ' ' not in cleaned_buffer and len(cleaned_buffer) > self.max_word_size:
                        # Force stream the long word
                        words_to_stream.append(cleaned_buffer)
                        self.buffer = ""
                    return words_to_stream
                
                # Special handling for newlines
                if '\n' in cleaned_buffer:
                    # Process up to the last newline to ensure we don't cut paragraphs
                    parts = cleaned_buffer.split('\n')
                    
                    # Keep the last part (after the last newline) in buffer if no space
                    last_part = parts[-1]
                    if ' ' not in last_part and len(parts) > 1:
                        # Stream all complete parts with newlines
                        to_stream = '\n'.join(parts[:-1]) + '\n'
                        words_to_stream.append(to_stream)
                        self.buffer = last_part
                        return words_to_stream
                
                # Normal word processing
                # Special regex that treats newlines as word separators like spaces
                parts = re.split(r'(\s+|\n+)', cleaned_buffer)
                
                words_streamed = []
                complete_pairs = (len(parts) // 2) * 2  # Ensure we only process complete word+separator pairs
                
                # Process word+separator pairs
                for i in range(0, complete_pairs, 2):
                    word = parts[i]
                    separator = parts[i + 1] if i + 1 < len(parts) else ""
                    
                    # Handle special case of newlines in the separator
                    if '\n' in separator:
                        # For newline separators, keep them with the word
                        complete_word = word + separator
                        words_streamed.append(complete_word)
                        words_to_stream.append(complete_word)
                    elif word:  # Only add non-empty words
                        complete_word = word + separator
                        words_streamed.append(complete_word)
                        words_to_stream.append(complete_word)
                
                # Remove streamed content from buffer
                if words_streamed:
                    streamed_text = ''.join(words_streamed)
                    self.buffer = self.buffer[len(streamed_text):]
                
                return words_to_stream
            
            def flush_remaining(self):
                """Flush any remaining content"""
                if self.buffer:
                    cleaned = clean_json_value(self.buffer)
                    self.buffer = ""
                    return [cleaned] if cleaned else []
                return []
        
        try:
            # Get context and model info asynchronously (EXACT same)
            cvs, provider_model = await prompt_service.get_cvs_and_provider_model(prompt_type)
            text = await context_service.get_context(document_id, cvs, variables or {}, prompt_type)
            
            accumulated_content = ""
            tag = input_tag if input_tag else uuid.uuid4().hex
            prompt = None
            
            # State tracking for real-time field detection (EXACT same)
            current_field = None
            field_value_buffer = ""
            expected_fields = ['selected_text_strategy', 'was_context_used', 'non_pasteable_answer_part', 'pasteable_answer_part']
            completed_fields = {}
            
            # Word buffering for each field to handle escape sequences properly
            field_buffers = {}
            
            # Check if this is script type (pasteable_answer_part will be array) (EXACT same)
            is_script_type = (variables and variables.get('type') == 'script') or (prompt_type in ["SCENE_IMPROVE_SELECTION_V2","SCENE_EXPAND_SELECTION_V2"]) 
            
            # REPLACE: Instead of external API call, use integrated streaming
            # Get prompt type and template for integrated streaming
            prompt_type_obj = PromptExecutorService.get_prompt_type(prompt_type)
            if not prompt_type_obj:
                error_msg = f'Prompt type "{prompt_type}" not configured'
                logger.error(f"❌ {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': error_msg})}" + "\n\n"
                return
            
            logger.info(f"✅ Found prompt type: {prompt_type}")
            logger.info(f"📋 JSON Schema: {prompt_type_obj.json_schema[:200] if prompt_type_obj.json_schema else 'None'}...")
            logger.info(f"🔧 System Prompt: {prompt_type_obj.system_prompt[:100] if prompt_type_obj.system_prompt else 'None'}...")
            
            # Get prompt template to access is_output_json flag
            _, _, encoded_template = PromptExecutorService.get_cvs_with_template(prompt_type)
            if not encoded_template:
                error_msg = f'Prompt template not found for "{prompt_type}"'
                logger.error(f"❌ {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': error_msg})}" + "\n\n"
                return
            
            # Decode the template
            import base64
            import pickle
            pickled_template = base64.b64decode(encoded_template)
            prompt_template = pickle.loads(pickled_template)
            
            # Parse provider and model
            if not provider_model:
                yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': 'LLM provider required'})}" + "\n\n"
                return
                
            try:
                provider, model_version = provider_model.split("_", 1)
            except ValueError:
                error_msg = f'Invalid provider_model format: {provider_model}'
                yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': error_msg})}" + "\n\n"
                return
            
            # Initialize GenAI service
            genai_service = GenAIService(provider=provider, model=model_version)
            logger.info(f"🤖 GenAI Service initialized: {provider}_{model_version}")
            
            # Build messages (same as prompt-executor)
            if prev_messages:
                if (prompt_type == "ASK_VED_PROMPT" or 
                    prompt_type == "ASK_VED_SCRIPT_PROMPT" or 
                    prompt_type.endswith("_EXPANSION") or 
                    prompt_type.endswith("_SUGGEST_NEXT")):
                    prev_messages.append({"role": "user", "content": variables.get('question', text)})
                else:
                    prev_messages.append({"role": "user", "content": text})
                messages = prev_messages
            else:
                messages = []
                if prompt_type_obj.system_prompt:
                    system_content = prompt_type_obj.system_prompt
                    if lang:
                        system_content += f"\\n - All the output should strictly be in {lang} language. If there is function calling or output is in json format then output accordingly such that the keys are in English and the values are in {lang}."
                    messages.append({"role": "system", "content": system_content})
                if text:
                    messages.append({"role": "user", "content": text})
            
            logger.info(f"📝 Messages built: {len(messages)} messages")
            logger.info(f"📤 Request to LLM - JSON Schema: {'Yes' if prompt_type_obj.json_schema else 'No'}")
            logger.info(f"📤 Request to LLM - Output JSON: {prompt_template.is_output_json}")
            logger.info(f"📤 Context length: {len(text)} characters")
            
            # Send initial metadata immediately as per frontend contract (EXACT same)
            yield f"data: {json.dumps({'type': 'start', 'field': 'prompt_type', 'data': prompt_type})}" + "\n\n"
            
            # Send tag immediately after prompt_type
            yield f"data: {json.dumps({'type': 'metadata', 'field': 'tag', 'data': tag})}" + "\n\n"
            
            # Process streaming response with integrated LLM (REPLACING external API call)
            async for llm_chunk in genai_service.llm_stream_async(
                messages, 
                json_schema=prompt_type_obj.json_schema,
                is_json_output=prompt_template.is_output_json,
                function_call=prompt_type_obj.function_output
            ):
                try:
                    # Handle different chunk types from LLM
                    if 'error' in llm_chunk:
                        yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': llm_chunk['error']})}" + "\n\n"
                        return
                    
                    if 'tag' in llm_chunk and not tag:
                        if input_tag and input_tag is not None:
                            tag = input_tag
                            yield f"data: {json.dumps({'type': 'metadata', 'field': 'tag', 'data': input_tag})}" + "\n\n"
                        else:
                            tag = llm_chunk['tag']
                            yield f"data: {json.dumps({'type': 'metadata', 'field': 'tag', 'data': tag})}" + "\n\n"

                    if 'prompt' in llm_chunk and not prompt:
                        prompt = llm_chunk['prompt']
                    
                    if 'content' in llm_chunk:
                        content = llm_chunk['content']
                        accumulated_content += content
                        logger.debug(f"📥 LLM Content: '{content[:50]}...'")
                        
                        # Process content for real-time field detection
                        for field_name in expected_fields:
                            if field_name not in completed_fields:
                                # Look for the field key pattern: "field_name": (handles both strings and arrays)
                                field_pattern = f'"{field_name}":'
                                if field_pattern in accumulated_content and current_field != field_name:
                                    # Found a new field! Send field_start
                                    logger.info(f"🔍 Found field pattern '{field_pattern}' in accumulated content")
                                    current_field = field_name
                                    field_value_buffer = ""
                                    yield f"data: {json.dumps({'type': 'field_start', 'field': field_name, 'data': ''})}" + "\n\n"
                                    await asyncio.sleep(0.001)
                                    break
                        
                        # If we're currently in a field, accumulate its value
                        if current_field:
                            # Look for field completion pattern
                            field_key_pattern = f'"{current_field}":'
                            if field_key_pattern in accumulated_content:
                                # Find where the field value starts
                                key_end_idx = accumulated_content.find(field_key_pattern) + len(field_key_pattern)
                                remaining_content = accumulated_content[key_end_idx:].lstrip()
                                
                                # Handle both string and array values
                                if remaining_content.startswith('"'):
                                    # String value: "field":"value"
                                    value_start_idx = key_end_idx + accumulated_content[key_end_idx:].find('"') + 1
                                    current_content_slice = accumulated_content[value_start_idx:]
                                    
                                    if len(current_content_slice) > len(field_value_buffer):
                                        new_content = current_content_slice[len(field_value_buffer):]
                                        field_value_buffer = current_content_slice
                                        
                                        if field_value_buffer.endswith('"') and not field_value_buffer.endswith('\\"'):
                                            # String field completed
                                            raw_value = field_value_buffer[:-1]
                                            final_chunk = new_content[:-1] if new_content.endswith('"') else new_content
                                            if final_chunk:
                                                # Initialize buffer for this field if needed
                                                if current_field not in field_buffers:
                                                    field_buffers[current_field] = StreamingWordBuffer(buffer_size=7, max_word_size=50)
                                                
                                                # Add final chunk and flush remaining content
                                                words_to_stream = field_buffers[current_field].add_content(final_chunk)
                                                remaining_words = field_buffers[current_field].flush_remaining()
                                                
                                                # Stream all chunks including remaining
                                                all_words = words_to_stream + remaining_words
                                                for word in all_words:
                                                    if word:
                                                        yield f"data: {json.dumps({'type': 'content', 'field': current_field, 'data': word})}" + "\n\n"
                                            cleaned_value = clean_json_value(raw_value)
                                            logger.info(f"🔍 Field '{current_field}' extracted: raw='{raw_value}' cleaned='{cleaned_value}' ({len(cleaned_value)} chars)")
                                            completed_fields[current_field] = cleaned_value
                                            current_field = None
                                            field_value_buffer = ""
                                        else:
                                            # Still streaming - use word buffering
                                            if new_content:
                                                # Initialize buffer for this field if needed
                                                if current_field not in field_buffers:
                                                    field_buffers[current_field] = StreamingWordBuffer(buffer_size=7, max_word_size=50)
                                                
                                                # Add content to buffer and get chunks to stream
                                                words_to_stream = field_buffers[current_field].add_content(new_content)
                                                
                                                # Stream each chunk
                                                for word in words_to_stream:
                                                    if word:  # Only stream non-empty words
                                                        yield f"data: {json.dumps({'type': 'content', 'field': current_field, 'data': word})}" + "\n\n"
                                                        await asyncio.sleep(0.01)  # Small delay for better UX
                                
                                elif remaining_content.startswith('[') and is_script_type and current_field == 'pasteable_answer_part':
                                    # Array value for script type pasteable_answer_part: "field":[...]
                                    array_start_idx = key_end_idx + accumulated_content[key_end_idx:].find('[')
                                    current_content_slice = accumulated_content[array_start_idx:]
                                    
                                    if len(current_content_slice) > len(field_value_buffer):
                                        field_value_buffer = current_content_slice
                                        
                                        if ']' in field_value_buffer and (field_value_buffer.endswith(']') or field_value_buffer.endswith(']}')):
                                            # Array field completed - convert to string and stream
                                            try:
                                                # Extract just the array part (remove trailing `}` if present)
                                                array_text = field_value_buffer
                                                if array_text.endswith(']}'):
                                                    array_text = array_text[:-1]  # Remove the trailing `}`
                                                elif array_text.endswith(']'):
                                                    # Already clean array
                                                    pass
                                                
                                                parsed_array = json.loads(array_text)
                                                if isinstance(parsed_array, list):
                                                    # Format as "element_type:element_text"
                                                    text_parts = []
                                                    for item in parsed_array:
                                                        if isinstance(item, dict) and 'element_text' in item and 'element_type' in item:
                                                            formatted_text = f"{item['element_type']}:{item['element_text']}"
                                                            text_parts.append(formatted_text)
                                                        elif isinstance(item, str):
                                                            text_parts.append(item)
                                                    string_value = ' \n '.join(text_parts)
                                                    # Stream the converted string value using word buffering
                                                    if current_field not in field_buffers:
                                                        field_buffers[current_field] = StreamingWordBuffer(buffer_size=7, max_word_size=50)
                                                    
                                                    # Add all content and flush
                                                    words_to_stream = field_buffers[current_field].add_content(string_value)
                                                    remaining_words = field_buffers[current_field].flush_remaining()
                                                    
                                                    # Stream all chunks
                                                    all_words = words_to_stream + remaining_words
                                                    for word in all_words:
                                                        if word:
                                                            yield f"data: {json.dumps({'type': 'content', 'field': current_field, 'data': word})}" + "\n\n"
                                                            await asyncio.sleep(0.02)
                                                    completed_fields[current_field] = clean_json_value(string_value)
                                                else:
                                                    completed_fields[current_field] = field_value_buffer
                                            except json.JSONDecodeError as e:
                                                logger.error(f"❌ Script array JSON error: {e}")
                                                completed_fields[current_field] = field_value_buffer
                                            
                                            current_field = None
                                            field_value_buffer = ""
                                    
                                    await asyncio.sleep(0.001)
                    
                    if 'final_output' in llm_chunk:
                        final_output = llm_chunk['final_output']
                        accumulated_content = final_output
                        logger.info(f"🔚 Final output received, length: {len(final_output)}")
                        logger.info(f"🔍 Final content preview: {final_output[:300]}...")
                        
                except Exception as chunk_error:
                    logger.error(f"Error processing LLM chunk: {str(chunk_error)}")
                    continue
        
            # Final processing and completion
            
            # Handle history updates for specific prompt types
            if (prompt_type == CONSTANTS.ASK_VED_PROMPT or 
                prompt_type == CONSTANTS.ASK_VED_SCRIPT_PROMPT or 
                prompt_type.endswith(CONSTANTS.EXPANSION_SUFFIX) or 
                prompt_type.endswith(CONSTANTS.SUGGEST_NEXT_SUFFIX)):
                
                final_tag = input_tag if input_tag else tag
                await update_ask_ved_history_async(final_tag, prompt, project_id, document_id)
            
            # Add tag to completed fields - use the same tag that was stored in history
            final_response_tag = input_tag if input_tag else tag
            if final_response_tag and 'tag' not in completed_fields:
                completed_fields['tag'] = final_response_tag
            
            # Handle final script type array conversion if not already processed (EXACT SAME)
            if is_script_type and 'pasteable_answer_part' not in completed_fields:
                # Try to parse the final accumulated_content as JSON to extract pasteable_answer_part
                try:
                    if accumulated_content.strip():
                        final_json = json.loads(accumulated_content.strip())
                        if isinstance(final_json, dict) and 'pasteable_answer_part' in final_json:
                            pasteable_value = final_json['pasteable_answer_part']
                            if isinstance(pasteable_value, list):
                                # Format as "element_type:element_text" with \n separator
                                text_parts = []
                                for item in pasteable_value:
                                    if isinstance(item, dict) and 'element_text' in item and 'element_type' in item:
                                        formatted_text = f"{item['element_type']}:{item['element_text']}"
                                        text_parts.append(formatted_text)
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                converted_value = ' \n '.join(text_parts)
                                cleaned_value = clean_json_value(converted_value)
                                logger.info(f"🔍 Array pasteable_answer_part: raw_array={pasteable_value} converted='{converted_value}' cleaned='{cleaned_value}' ({len(cleaned_value)} chars)")
                                completed_fields['pasteable_answer_part'] = cleaned_value
                            else:
                                cleaned_value = clean_json_value(pasteable_value)
                                logger.info(f"🔍 String pasteable_answer_part: raw='{pasteable_value}' cleaned='{cleaned_value}' ({len(cleaned_value)} chars)")
                                completed_fields['pasteable_answer_part'] = cleaned_value
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"⚠️ Failed to process final script response: {e}")
            
            # Flush any remaining content in field buffers before final result (EXACT SAME)
            for field_name, buffer in field_buffers.items():
                remaining_words = buffer.flush_remaining()
                for word in remaining_words:
                    if word:
                        yield f"data: {json.dumps({'type': 'content', 'field': field_name, 'data': word})}" + "\n\n"
        
            # Clean all field values before sending final result (EXACT SAME)
            logger.info(f"🧹 Completed fields before cleaning: {list(completed_fields.keys())}")
            for field_name in expected_fields:
                if field_name not in completed_fields:
                    logger.warning(f"⚠️ Missing expected field: {field_name}")
                else:
                    logger.info(f"✅ Found field '{field_name}': {len(str(completed_fields[field_name]))} chars")
            
            cleaned_final_fields = {}
            for key, value in completed_fields.items():
                cleaned_value = clean_json_value(value)
                logger.info(f"🧹 Final cleaning '{key}': before='{value}' after='{cleaned_value}' ({len(cleaned_value)} chars)")
                cleaned_final_fields[key] = cleaned_value

            
            # Send end signal (EXACT SAME)
            yield f"data: {json.dumps({'type': 'end', 'field': 'final', 'data': cleaned_final_fields})}" + "\n\n"
            
        except Exception as e:
            logger.error(f"ASYNC streaming execution failed: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'field': '', 'data': str(e)})}" + "\n\n"

    # Return async streaming response (EXACT same headers as core-backend)
    return StreamingResponse(
        async_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0", 
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept, Cache-Control, Connection, Authorization",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Transfer-Encoding": "chunked"
        }
    )