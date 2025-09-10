"""MongoDB connection and utilities"""
import motor.motor_asyncio
from app.core.config import settings

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    mongodb.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_HOST)
    mongodb.database = mongodb.client[settings.MONGO_DB]

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()

def get_database():
    """Get database instance"""
    return mongodb.database