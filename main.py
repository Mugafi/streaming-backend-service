from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import health, main as main_router, execute_stream
from app.core.config import settings
from app.core.mongodb import connect_to_mongo, close_mongo_connection

app = FastAPI(
    title="Streaming Backend Service",
    description="FastAPI service for streaming functionality, migrated from Django's execute-async-stream API",
    version="1.0.0",
)

# MongoDB connection events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(main_router.router, prefix="", tags=["main"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(execute_stream.router, prefix="/v1/copilot", tags=["streaming"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)