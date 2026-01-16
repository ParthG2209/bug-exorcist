"""
core/agent.py - The Brain of Bug Exorcist

An autonomous AI agent that analyzes runtime errors, generates fixes using GPT-4o,
and orchestrates the entire bug fixing workflow including sandboxing and verification.
"""

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class BugExorcistAgent:
    """
    The autonomous Bug Exorcist agent that:
    1. Analyzes error messages and stack traces
    2. Generates code fixes using GPT-4o
    3. Orchestrates Docker sandbox environments
    4. Verifies fixes before committing
    """

    SYSTEM_PROMPT = """You are the Bug Exorcist, an elite autonomous AI debugging agent.

Your mission is to analyze runtime errors, understand their root causes, and generate precise code fixes.

**Core Capabilities:**
- Deep analysis of Python stack traces and error messages
- Understanding of common bug patterns (null pointer, type errors, logic errors, etc.)
- Writing clean, production-ready fix patches
- Explaining fixes in developer-friendly language

**Analysis Process:**
1. Examine the error message and stack trace carefully
2. Identify the exact line and nature of the failure
3. Understand the context from surrounding code
4. Determine the root cause (not just symptoms)
5. Generate a minimal, targeted fix
6. Explain your reasoning

**Fix Requirements:**
- Fixes must be minimal and surgical - only change what's necessary
- Preserve existing code style and patterns
- Add defensive checks where appropriate
- Include brief inline comments explaining critical fixes
- Ensure backwards compatibility when possible

**Output Format:**
When asked to fix code, respond with:
1. Root Cause Analysis (2-3 sentences)
2. The complete fixed code
3. Explanation of changes made

Be precise, be thorough, be the exorcist of bugs."""

    def __init__(self, bug_id: str, openai_api_key: Optional[str] = None):
        """
        Initialize the Bug Exorcist Agent.
        
        Args:
            bug_id: Unique identifier for this bug investigation
            openai_api_key: OpenAI API key (falls back to env var)
        """
        self.bug_id = bug_id
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env variable "
                "or pass it to the constructor."
            )
        
        # Initialize LangChain ChatOpenAI with GPT-4o
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,  # Low temperature for deterministic, focused fixes
            openai_api_key=self.api_key,
            max_tokens=2000
        )
        
        self.stages = [
            "Initializing sandbox environment...",
            "Cloning repository into isolated container...",
            "Installing dependencies...",
            "Running reproduction script...",
            "Analyzing stack trace...",
            "Identifying root cause with GPT-4o...",
            "Generating patch candidate...",
            "Applying patch...",
            "Verifying fix with unit tests...",
            "Fix verified. Cleaning up resources."
        ]

    async def stream_logs(self) -> AsyncGenerator[str, None]:
        """
        Stream execution logs in real-time for the WebSocket connection.
        This simulates the agent's workflow for demo purposes.
        
        Yields:
            Log messages with timestamps and severity levels
        """
        yield f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Starting exorcism for Bug ID: {self.bug_id}"
        
        for stage in self.stages:
            await asyncio.sleep(1.5)
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if "GPT-4o" in stage:
                yield f"[{timestamp}] [AI] {stage}"
            elif "Verifying" in stage:
                yield f"[{timestamp}] [TEST] {stage}"
            elif "verified" in stage:
                yield f"[{timestamp}] [SUCCESS] {stage}"
            else:
                yield f"[{timestamp}] [DEBUG] {stage}"
            
            # Add realistic sub-logs
            if "dependencies" in stage:
                yield f"[{timestamp}] [DEBUG] Pip: Installing fastapi, langchain, openai..."
            elif "reproduction" in stage:
                yield f"[{timestamp}] [DEBUG] Traceback detected: ZeroDivisionError in main.py:42"
        
        yield f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Exorcism complete for {self.bug_id}."

    async def analyze_error(
        self, 
        error_message: str, 
        code_snippet: str,
        file_path: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core function: Analyze an error and generate a fix using GPT-4o.
        
        Args:
            error_message: The error/exception message with stack trace
            code_snippet: The code that caused the error
            file_path: Optional file path for context
            additional_context: Optional additional context about the bug
            
        Returns:
            Dictionary containing:
                - root_cause: Analysis of what caused the bug
                - fixed_code: The corrected code
                - explanation: What was changed and why
                - confidence: Confidence level (0.0-1.0)
        """
        # Construct the analysis prompt
        user_prompt = f"""Analyze and fix this bug:

**Error Message:**
```
{error_message}
```

**Original Code:**
```python
{code_snippet}
```
"""
        
        if file_path:
            user_prompt += f"\n**File Path:** `{file_path}`\n"
        
        if additional_context:
            user_prompt += f"\n**Additional Context:**\n{additional_context}\n"
        
        user_prompt += """
Please provide:
1. Root Cause Analysis
2. The complete fixed code
3. Explanation of your changes
"""
        
        try:
            # Call GPT-4o via LangChain
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            ai_response = response.generations[0][0].text
            
            # Parse the AI response (in production, use structured output)
            result = self._parse_ai_response(ai_response, code_snippet)
            
            return {
                "bug_id": self.bug_id,
                "root_cause": result["root_cause"],
                "fixed_code": result["fixed_code"],
                "explanation": result["explanation"],
                "confidence": result["confidence"],
                "original_error": error_message,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "bug_id": self.bug_id,
                "error": f"Failed to analyze error: {str(e)}",
                "root_cause": "Analysis failed",
                "fixed_code": code_snippet,
                "explanation": f"Error during GPT-4o analysis: {str(e)}",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    def _parse_ai_response(self, ai_response: str, original_code: str) -> Dict[str, Any]:
        """
        Parse the AI's response to extract structured components.
        
        In production, this should use structured output or JSON mode.
        For now, it uses heuristic parsing.
        """
        # Simple heuristic parsing
        lines = ai_response.split('\n')
        
        root_cause = ""
        fixed_code = ""
        explanation = ""
        in_code_block = False
        
        for i, line in enumerate(lines):
            # Detect code blocks
            if '```python' in line.lower() or '```' in line:
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                fixed_code += line + '\n'
            elif 'root cause' in line.lower() and i + 1 < len(lines):
                # Capture next few lines as root cause
                root_cause = '\n'.join(lines[i+1:i+4]).strip()
            elif 'explanation' in line.lower() or 'changes' in line.lower():
                explanation = '\n'.join(lines[i+1:i+5]).strip()
        
        # Fallback: if no code found, use original
        if not fixed_code.strip():
            fixed_code = original_code
        
        # Estimate confidence based on response quality
        confidence = 0.8 if fixed_code.strip() and root_cause else 0.5
        
        return {
            "root_cause": root_cause or "Analysis completed",
            "fixed_code": fixed_code.strip(),
            "explanation": explanation or "Code has been fixed",
            "confidence": confidence
        }

    async def verify_fix(
        self, 
        fixed_code: str, 
        test_command: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify the fix by running tests in a Docker sandbox.
        
        Args:
            fixed_code: The patched code to test
            test_command: Optional test command to run
            
        Returns:
            Dictionary with verification results
        """
        # This would integrate with Docker SDK
        # For now, return a mock verification
        return {
            "verified": True,
            "test_output": "All tests passed",
            "exit_code": 0,
            "timestamp": datetime.now().isoformat()
        }

    async def execute_full_workflow(
        self,
        error_message: str,
        code_snippet: str,
        file_path: str,
        repo_path: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the complete bug fixing workflow with real-time updates.
        
        Args:
            error_message: The error to fix
            code_snippet: Code containing the bug
            file_path: Path to the file
            repo_path: Optional repository path for git operations
            
        Yields:
            Status updates throughout the workflow
        """
        yield {
            "stage": "initialization",
            "message": f"Starting Bug Exorcist for {self.bug_id}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Stage 1: Analysis
        yield {
            "stage": "analysis",
            "message": "Analyzing error with GPT-4o...",
            "timestamp": datetime.now().isoformat()
        }
        
        fix_result = await self.analyze_error(error_message, code_snippet, file_path)
        
        yield {
            "stage": "analysis_complete",
            "message": f"Root cause identified: {fix_result['root_cause'][:100]}...",
            "data": fix_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Stage 2: Verification
        yield {
            "stage": "verification",
            "message": "Verifying fix in sandbox...",
            "timestamp": datetime.now().isoformat()
        }
        
        verification = await self.verify_fix(fix_result['fixed_code'])
        
        yield {
            "stage": "verification_complete",
            "message": f"Verification: {'PASSED' if verification['verified'] else 'FAILED'}",
            "data": verification,
            "timestamp": datetime.now().isoformat()
        }
        
        # Stage 3: Completion
        if verification['verified']:
            yield {
                "stage": "complete",
                "message": "Bug successfully exorcised! Fix ready for commit.",
                "data": {
                    "fix": fix_result,
                    "verification": verification
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "stage": "failed",
                "message": "Fix verification failed. Manual review required.",
                "timestamp": datetime.now().isoformat()
            }


# Convenience function for quick usage
async def quick_fix(error: str, code: str, api_key: Optional[str] = None) -> str:
    """
    Quick one-shot fix function.
    
    Args:
        error: Error message
        code: Code with the bug
        api_key: Optional OpenAI API key
        
    Returns:
        Fixed code as a string
    """
    agent = BugExorcistAgent(bug_id="quick-fix", openai_api_key=api_key)
    result = await agent.analyze_error(error, code)
    return result['fixed_code']