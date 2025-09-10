"""Request and response models for prompt management"""
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ExecuteStreamRequest(BaseModel):
    """Request model for execute async stream endpoint"""
    customer_id: Optional[str] = None
    document_id: str
    lang: Optional[str] = None
    stream: Optional[bool] = True
    prompt_type: str
    placeholders: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None
    tag: Optional[str] = None
    page_num: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "675006c0ef79d1017de4f5f2",
                "document_id": "68b18b30551c65b7404e28bb",
                "lang": "English",
                "stream": True,
                "prompt_type": "ASK_QN_VED_V2",
                "placeholders": {
                    "selected": "",
                    "type": "idea",
                    "question": "give me random idea of story under 300 words"
                },
                "page_num": 1,
                "tag": None
            }
        }