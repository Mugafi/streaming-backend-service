"""Application constants - EXACT copy from core-backend"""

# Default model configuration
DEFAULT_LLM_MODEL = "openai_gpt-3.5-turbo-1106"
DEFAULT_PROVIDER = "openai"

# HTTP status codes
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_500_INTERNAL_SERVER_ERROR = 500

# Field names for streaming
STREAMING_FIELDS = [
    'selected_text_strategy',
    'was_context_used', 
    'non_pasteable_answer_part',
    'pasteable_answer_part'
]

# Buffer settings
DEFAULT_BUFFER_SIZE = 7
MAX_WORD_SIZE = 50

# Scene prompt type mappings
SCENE_PROMPT_TYPE_OBJECT_TYPE = {
    "SCENE_SUGGEST_NEXT_NEW": 'next_suggested_entity', 
    "SCENE_EXPANSION_NEW": "expanded_elements",
    "SCENE_IMPROVEMENT_NEW": "improved_elements",
    "SCENE_SUGGEST_NEXT_V2": "pasteable_answer_part",
    "SCENE_IMPROVE_SELECTION_V2": "pasteable_answer_part",
    "SCENE_EXPAND_SELECTION_V2": "pasteable_answer_part",
    "ASK_QN_VED_SCRIPT_V2": "pasteable_answer_part",
}

class CONSTANTS:
    RESEARCH_INIT_QUESTIONS_PROMPT = "RESEARCH_INIT_QUESTIONS"
    ASK_VED_PROMPT = "ASK_QN_VED_V2"
    ASK_VED_SCRIPT_PROMPT = "ASK_QN_VED_SCRIPT_V2"
    EXPANSION_SUFFIX = "_EXPANSION"
    SUGGEST_NEXT_SUFFIX = "_SUGGEST_NEXT"