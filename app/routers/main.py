"""Main router for root endpoints"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Streaming Backend Service",
        "description": "FastAPI service for streaming functionality, migrated from Django's execute-async-stream API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "streaming": "/v1/copilot/execute-async-stream"
        }
    }