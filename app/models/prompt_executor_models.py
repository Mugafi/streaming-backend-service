"""Database models for prompt executor functionality - EXACT copy of Django models"""
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class PromptTypes(Base):
    """Prompt types table - matches Django PromptTypes model exactly"""
    __tablename__ = "prompt_types"
    
    name = Column(String(64), primary_key=True)
    system_prompt = Column(Text, default=None, nullable=True)
    function_output = Column(JSON, default=dict, nullable=True)
    json_schema = Column(Text, default=None, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class GenAIModelPromptTypeMapping(Base):
    """Mapping between GenAI models and prompt types - matches Django model exactly"""
    __tablename__ = "genai_model_prompt_type_mapping"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_type_id = Column(String(64), ForeignKey("prompt_types.name"), nullable=False)
    genai_model = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class PromptTemplate(Base):
    """Prompt templates table - matches Django PromptTemplate model exactly"""
    __tablename__ = "prompt_template"
    
    id = Column(String(36), primary_key=True)
    prompt = Column(Text, nullable=False)
    prompt_type_id = Column(String(64), ForeignKey("prompt_types.name"), nullable=False)
    contains_placeholder = Column(Boolean, default=False)
    placeholders = Column(String(200), default=None, nullable=True)
    iterations_required = Column(Boolean, default=False)
    major_version = Column(Integer, default=1)
    minor_version = Column(Integer, default=0)
    prompt_sequence = Column(Integer, default=1)
    is_output_json = Column(Boolean, default=False)
    context_variables = Column(String(250), default=None, nullable=True)
    is_context_mandatory = Column(Boolean, default=False)
    output_format = Column(Text, default=None, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())