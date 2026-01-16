"""
backend/app/main.py - Main FastAPI Application

Enhanced with Bug Exorcist Agent integration.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.logs import router as logs_router
from app.api.agent import router as agent_router
from app.database import engine, Base

# Load environment variables from .env file
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Bug Exorcist API",
    description="Autonomous AI-powered debugging system with GPT-4o integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "active",
        "service": "Bug Exorcist",
        "version": "1.0.0",
        "features": [
            "autonomous_debugging",
            "gpt4o_integration",
            "docker_sandboxing",
            "git_operations",
            "websocket_logging"
        ]
    }

# Configure CORS (Essential for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(logs_router, tags=["logs"])
app.include_router(agent_router, tags=["agent"])

@app.get("/")
async def root():
    return {
        "message": "🧟‍♂️ Bug Exorcist API is running",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze_bug": "POST /api/agent/analyze",
            "quick_fix": "POST /api/agent/quick-fix",
            "agent_health": "GET /api/agent/health",
            "websocket_logs": "WS /ws/logs/{bug_id}"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🧟‍♂️ Bug Exorcist API Starting...")
    print("=" * 60)
    print("📡 WebSocket logging: /ws/logs/{bug_id}")
    print("🤖 Agent analysis: POST /api/agent/analyze")
    print("⚡ Quick fix: POST /api/agent/quick-fix")
    print("📚 Documentation: http://localhost:8000/docs")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("🧟‍♂️ Bug Exorcist API shutting down...")