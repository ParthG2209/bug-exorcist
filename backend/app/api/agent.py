"""
backend/app/api/agent.py - FastAPI endpoints for Bug Exorcist Agent

This module provides REST API endpoints to interact with the autonomous debugging agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
import os

from core.agent import BugExorcistAgent, quick_fix
from app.database import SessionLocal
from app import crud


router = APIRouter(prefix="/api/agent", tags=["agent"])


# Request/Response Models
class BugAnalysisRequest(BaseModel):
    """Request model for bug analysis"""
    error_message: str = Field(..., description="The error message with stack trace")
    code_snippet: str = Field(..., description="The code that caused the error")
    file_path: Optional[str] = Field(None, description="Path to the file containing the bug")
    additional_context: Optional[str] = Field(None, description="Additional context about the bug")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key (optional, uses env if not provided)")

    class Config:
        json_schema_extra = {
            "example": {
                "error_message": "ZeroDivisionError: division by zero\n  File 'calc.py', line 10",
                "code_snippet": "def divide(a, b):\n    return a / b",
                "file_path": "calc.py",
                "additional_context": "This function is called from the API endpoint"
            }
        }


class BugAnalysisResponse(BaseModel):
    """Response model for bug analysis"""
    bug_id: str
    root_cause: str
    fixed_code: str
    explanation: str
    confidence: float
    original_error: str
    timestamp: str


class QuickFixRequest(BaseModel):
    """Request model for quick fix"""
    error: str
    code: str
    openai_api_key: Optional[str] = None


class QuickFixResponse(BaseModel):
    """Response model for quick fix"""
    fixed_code: str


# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/analyze", response_model=BugAnalysisResponse)
async def analyze_bug(request: BugAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyze a bug and generate a fix using GPT-4o.
    
    This endpoint:
    1. Creates a bug report in the database
    2. Analyzes the error using the Bug Exorcist agent
    3. Returns the fix with explanation
    
    Args:
        request: Bug analysis request with error details
        db: Database session
        
    Returns:
        Analysis results with fixed code
    """
    try:
        # Create bug report in database
        bug_report = crud.create_bug_report(
            db=db,
            description=f"{request.error_message[:200]}..."
        )
        bug_id = f"BUG-{bug_report.id}"
        
        # Get API key (from request or environment)
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required. Provide it in the request or set OPENAI_API_KEY environment variable."
            )
        
        # Initialize agent
        agent = BugExorcistAgent(bug_id=bug_id, openai_api_key=api_key)
        
        # Analyze the error
        result = await agent.analyze_error(
            error_message=request.error_message,
            code_snippet=request.code_snippet,
            file_path=request.file_path,
            additional_context=request.additional_context
        )
        
        # Update bug report status
        crud.update_bug_report_status(db=db, bug_report_id=bug_report.id, status="analyzed")
        
        return BugAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/quick-fix", response_model=QuickFixResponse)
async def quick_fix_endpoint(request: QuickFixRequest):
    """
    Quick fix endpoint - returns only the fixed code without full analysis.
    
    Useful for simple, fast fixes where you don't need detailed explanations.
    
    Args:
        request: Quick fix request with error and code
        
    Returns:
        Only the fixed code
    """
    try:
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required"
            )
        
        fixed = await quick_fix(
            error=request.error,
            code=request.code,
            api_key=api_key
        )
        
        return QuickFixResponse(fixed_code=fixed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick fix failed: {str(e)}")


@router.get("/health")
async def agent_health():
    """
    Check if the agent system is operational.
    
    Returns:
        Health status and configuration info
    """
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    
    return {
        "status": "operational",
        "agent": "Bug Exorcist",
        "model": "gpt-4o",
        "api_key_configured": api_key_set,
        "langchain_available": True,
        "capabilities": [
            "error_analysis",
            "code_fixing",
            "root_cause_detection",
            "automated_verification"
        ]
    }


@router.get("/bugs/{bug_id}/status")
async def get_bug_status(bug_id: int, db: Session = Depends(get_db)):
    """
    Get the status of a bug report.
    
    Args:
        bug_id: ID of the bug report
        db: Database session
        
    Returns:
        Bug report details
    """
    bug_report = crud.get_bug_report(db, bug_id)
    
    if not bug_report:
        raise HTTPException(status_code=404, detail="Bug not found")
    
    return {
        "id": bug_report.id,
        "description": bug_report.description,
        "status": bug_report.status,
        "created_at": bug_report.created_at.isoformat()
    }


@router.get("/bugs")
async def list_bugs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List all bug reports.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of bug reports
    """
    bugs = crud.get_bug_reports(db, skip=skip, limit=limit)
    
    return {
        "bugs": [
            {
                "id": bug.id,
                "description": bug.description,
                "status": bug.status,
                "created_at": bug.created_at.isoformat()
            }
            for bug in bugs
        ],
        "count": len(bugs)
    }


@router.post("/bugs/{bug_id}/verify")
async def verify_bug_fix(bug_id: int, fixed_code: str, db: Session = Depends(get_db)):
    """
    Verify a bug fix by running it in a sandbox.
    
    Args:
        bug_id: ID of the bug report
        fixed_code: The fixed code to verify
        db: Database session
        
    Returns:
        Verification results
    """
    bug_report = crud.get_bug_report(db, bug_id)
    
    if not bug_report:
        raise HTTPException(status_code=404, detail="Bug not found")
    
    # Initialize agent
    agent = BugExorcistAgent(bug_id=f"BUG-{bug_id}")
    
    # Verify the fix
    verification = await agent.verify_fix(fixed_code)
    
    # Update status if verified
    if verification['verified']:
        crud.update_bug_report_status(db=db, bug_report_id=bug_id, status="verified")
    else:
        crud.update_bug_report_status(db=db, bug_report_id=bug_id, status="verification_failed")
    
    return verification


@router.post("/test-connection")
async def test_openai_connection(api_key: Optional[str] = None):
    """
    Test the OpenAI API connection.
    
    Args:
        api_key: Optional API key to test (uses env if not provided)
        
    Returns:
        Connection test results
    """
    test_key = api_key or os.getenv("OPENAI_API_KEY")
    
    if not test_key:
        return {
            "success": False,
            "error": "No API key provided"
        }
    
    try:
        # Simple test with a minimal agent
        agent = BugExorcistAgent(bug_id="test", openai_api_key=test_key)
        
        # Try a simple analysis
        test_result = await agent.analyze_error(
            error_message="Test error",
            code_snippet="print('test')"
        )
        
        return {
            "success": True,
            "message": "OpenAI connection successful",
            "model": "gpt-4o",
            "test_confidence": test_result.get('confidence', 0.0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }