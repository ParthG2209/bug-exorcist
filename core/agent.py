"""core/agent.py - The Brain of Bug Exorcist

An autonomous AI agent that analyzes runtime errors, generates fixes using GPT-4o,
and orchestrates the entire bug fixing workflow including sandboxing and verification.

Enhanced with:
- Automatic retry logic (max 3 attempts)
- Gemini AI fallback when GPT-4o fails
- Graceful fallback to manual guidance
"""

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import fallback handlers
from core.fallback import get_fallback_handler
from core.gemini_agent import GeminiFallbackAgent, is_gemini_enabled, is_gemini_available


class BugExorcistAgent:
    """
    The autonomous Bug Exorcist agent that:
    1. Analyzes error messages and stack traces using GPT-4o
    2. Generates code fixes
    3. Orchestrates Docker sandbox environments
    4. Verifies fixes before committing
    5. Automatically retries up to 3 times if a fix fails
    6. Falls back to Gemini AI if GPT-4o fails
    7. Provides graceful manual guidance when all AI attempts fail
    """

    SYSTEM_PROMPT = """You are the Bug Exorcist, an elite autonomous AI debugging agent.

Your mission is to analyze runtime errors, understand their root causes, and generate precise code fixes.

**Core Capabilities:**
- Deep analysis of Python stack traces and error messages
- Understanding of common bug patterns (null pointer, type errors, logic errors, etc.)
- Writing clean, production-ready fix patches
- Explaining fixes in developer-friendly language
- Learning from failed fix attempts to generate better solutions

**Analysis Process:**
1. Examine the error message and stack trace carefully
2. Identify the exact line and nature of the failure
3. Understand the context from surrounding code
4. Determine the root cause (not just symptoms)
5. Generate a minimal, targeted fix
6. Explain your reasoning

**Retry Logic:**
When a fix fails verification, you will receive:
- The original error
- Your previous fix attempt
- The new error that occurred
- Attempt number (1-3)

Use this information to:
- Identify why the previous fix failed
- Avoid repeating the same mistake
- Generate a more robust solution
- Consider edge cases missed in previous attempts

**Fix Requirements:**
- Fixes must be minimal and surgical - only change what's necessary
- Preserve existing code style and patterns
- Add defensive checks where appropriate
- Include brief inline comments explaining critical fixes
- Ensure backwards compatibility when possible
- Learn from previous failed attempts

**Output Format:**
When asked to fix code, respond with:
1. Root Cause Analysis (2-3 sentences)
2. The complete fixed code
3. Explanation of changes made
4. (On retry) What was wrong with the previous attempt

Be precise, be thorough, be the exorcist of bugs."""

    MAX_RETRY_ATTEMPTS = 3  # Maximum number of fix attempts

    def __init__(self, bug_id: str, openai_api_key: Optional[str] = None):
        """
        Initialize the Bug Exorcist Agent.
        
        Args:
            bug_id: Unique identifier for this bug investigation
            openai_api_key: OpenAI API key (falls back to env var)
        """
        self.bug_id = bug_id
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Get fallback handler
        self.fallback_handler = get_fallback_handler()
        
        # Initialize Gemini fallback agent if enabled
        self.gemini_agent = None
        if is_gemini_enabled() and is_gemini_available():
            try:
                self.gemini_agent = GeminiFallbackAgent()
                print(f"[AGENT] Gemini fallback agent initialized and ready")
            except Exception as e:
                print(f"[AGENT] Warning: Could not initialize Gemini agent: {e}")
        
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
        
        # Track retry history
        self.retry_history: List[Dict[str, Any]] = []
        
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
        additional_context: Optional[str] = None,
        previous_attempts: Optional[List[Dict[str, Any]]] = None,
        use_gemini: bool = False
    ) -> Dict[str, Any]:
        """
        Core function: Analyze an error and generate a fix using AI.
        
        Args:
            error_message: The error/exception message with stack trace
            code_snippet: The code that caused the error
            file_path: Optional file path for context
            additional_context: Optional additional context about the bug
            previous_attempts: List of previous fix attempts that failed
            use_gemini: If True, use Gemini instead of GPT-4o
            
        Returns:
            Dictionary containing analysis results
        """
        attempt_number = len(previous_attempts) + 1 if previous_attempts else 1
        
        # If Gemini is requested but not available, fall back to GPT-4o
        if use_gemini and not self.gemini_agent:
            print("[AGENT] Gemini requested but not available, using GPT-4o")
            use_gemini = False
        
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
        
        # Add retry context if this is not the first attempt
        if previous_attempts:
            user_prompt += f"\n**RETRY ATTEMPT #{attempt_number}**\n"
            user_prompt += f"**Previous attempts have failed. Learn from these mistakes:**\n\n"
            
            for i, attempt in enumerate(previous_attempts, 1):
                user_prompt += f"--- Attempt {i} ---\n"
                user_prompt += f"**Fix Attempted:**\n```python\n{attempt['fixed_code']}\n```\n"
                user_prompt += f"**Result:** {attempt['verification_result']}\n"
                if attempt.get('new_error'):
                    user_prompt += f"**New Error:** {attempt['new_error']}\n"
                user_prompt += "\n"
            
            user_prompt += f"""
**IMPORTANT:** 
- Analyze why the previous fix(es) failed
- Do NOT repeat the same approach
- Generate a MORE ROBUST solution that addresses the failures
- Consider edge cases that were missed
"""
        
        user_prompt += """
Please provide:
1. Root Cause Analysis
2. The complete fixed code
3. Explanation of your changes
"""
        
        if previous_attempts:
            user_prompt += "4. What was wrong with the previous attempt(s) and how this fix is different\n"
        
        try:
            if use_gemini:
                # Use Gemini fallback agent
                print(f"[AGENT] Using Gemini AI for analysis (attempt {attempt_number})...")
                return await self.gemini_agent.analyze_error(
                    error_message=error_message,
                    code_snippet=code_snippet,
                    file_path=file_path,
                    additional_context=additional_context,
                    previous_attempts=previous_attempts
                )
            else:
                # Use GPT-4o
                print(f"[AGENT] Using GPT-4o for analysis (attempt {attempt_number})...")
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await self.llm.agenerate([messages])
                ai_response = response.generations[0][0].text
                
                # Parse the AI response
                result = self._parse_ai_response(ai_response, code_snippet)
                
                return {
                    "bug_id": self.bug_id,
                    "ai_agent": "gpt-4o",
                    "fallback_agent": False,
                    "root_cause": result["root_cause"],
                    "fixed_code": result["fixed_code"],
                    "explanation": result["explanation"],
                    "confidence": result["confidence"],
                    "original_error": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "attempt_number": attempt_number,
                    "retry_analysis": result.get("retry_analysis", "")
                }
            
        except Exception as e:
            # GPT-4o failed - try Gemini if available and not already using it
            if not use_gemini and self.gemini_agent:
                print(f"[AGENT] ⚠️ GPT-4o failed: {str(e)}")
                print(f"[AGENT] 🔄 Switching to Gemini AI fallback...")
                
                try:
                    gemini_result = await self.gemini_agent.analyze_error(
                        error_message=error_message,
                        code_snippet=code_snippet,
                        file_path=file_path,
                        additional_context=additional_context,
                        previous_attempts=previous_attempts,
                        gpt_failure_context=f"GPT-4o failed with error: {str(e)}"
                    )
                    gemini_result['bug_id'] = self.bug_id
                    return gemini_result
                    
                except Exception as gemini_error:
                    print(f"[AGENT] ❌ Gemini also failed: {str(gemini_error)}")
                    # Both AI systems failed - use manual fallback
                    if self.fallback_handler.is_enabled():
                        return self.fallback_handler.generate_api_failure_response(
                            error_message=error_message,
                            bug_id=self.bug_id,
                            api_error=f"GPT-4o: {str(e)}, Gemini: {str(gemini_error)}"
                        )
            
            # If API fails and no Gemini available, use fallback if enabled
            if self.fallback_handler.is_enabled():
                return self.fallback_handler.generate_api_failure_response(
                    error_message=error_message,
                    bug_id=self.bug_id,
                    api_error=str(e)
                )
            else:
                return {
                    "bug_id": self.bug_id,
                    "error": f"Failed to analyze error: {str(e)}",
                    "root_cause": "Analysis failed",
                    "fixed_code": code_snippet,
                    "explanation": f"Error during AI analysis: {str(e)}",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "attempt_number": attempt_number
                }

    def _parse_ai_response(self, ai_response: str, original_code: str) -> Dict[str, Any]:
        """
        Parse the AI's response to extract structured components.
        
        In production, this should use structured output or JSON mode.
        For now, it uses heuristic parsing.
        """
        lines = ai_response.split('\n')
        
        root_cause = ""
        fixed_code = ""
        explanation = ""
        retry_analysis = ""
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
            elif 'wrong with' in line.lower() or 'previous attempt' in line.lower():
                retry_analysis = '\n'.join(lines[i:i+4]).strip()
        
        # Fallback: if no code found, use original
        if not fixed_code.strip():
            fixed_code = original_code
        
        # Estimate confidence based on response quality
        confidence = 0.8 if fixed_code.strip() and root_cause else 0.5
        
        return {
            "root_cause": root_cause or "Analysis completed",
            "fixed_code": fixed_code.strip(),
            "explanation": explanation or "Code has been fixed",
            "confidence": confidence,
            "retry_analysis": retry_analysis
        }

    async def verify_fix(
        self, 
        fixed_code: str, 
        test_command: Optional[str] = None,
        original_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify the fix by running tests in a Docker sandbox.
        
        Args:
            fixed_code: The patched code to test
            test_command: Optional test command to run
            original_error: The original error for comparison
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Import sandbox here to avoid circular imports
            from app.sandbox import Sandbox
            
            sandbox = Sandbox()
            
            # Run the fixed code
            result = sandbox.run_code(fixed_code)
            
            # Check if execution was successful
            verified = not ("Error" in result or "Traceback" in result)
            
            return {
                "verified": verified,
                "test_output": result,
                "exit_code": 0 if verified else 1,
                "new_error": result if not verified else None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "verified": False,
                "test_output": f"Verification error: {str(e)}",
                "exit_code": 1,
                "new_error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_and_fix_with_retry(
        self,
        error_message: str,
        code_snippet: str,
        file_path: Optional[str] = None,
        additional_context: Optional[str] = None,
        max_attempts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze and fix a bug with automatic retry logic and AI fallback.
        
        This method will:
        1. Attempt to fix using GPT-4o (up to MAX_RETRY_ATTEMPTS)
        2. If GPT-4o fails, switch to Gemini AI
        3. If Gemini also fails after retries, use manual fallback
        
        Args:
            error_message: The error to fix
            code_snippet: Code containing the bug
            file_path: Optional file path
            additional_context: Optional additional context
            max_attempts: Override default max attempts (default: 3)
            
        Returns:
            Dictionary containing fix results and fallback information
        """
        max_attempts = max_attempts or self.MAX_RETRY_ATTEMPTS
        all_attempts = []
        gpt_failed = False
        using_gemini = False
        
        for attempt_num in range(1, max_attempts + 1):
            print(f"[AGENT] Attempt {attempt_num}/{max_attempts} - Analyzing bug...")
            
            try:
                # Generate fix (with Gemini if GPT-4o has failed)
                fix_result = await self.analyze_error(
                    error_message=error_message,
                    code_snippet=code_snippet,
                    file_path=file_path,
                    additional_context=additional_context,
                    previous_attempts=all_attempts,
                    use_gemini=using_gemini
                )
                
                # Check if this was an API failure that triggered fallback
                if fix_result.get("status") == "api_connection_failed":
                    print(f"[AGENT] ❌ API connection failed. Returning fallback response.")
                    return {
                        "success": False,
                        "final_fix": None,
                        "fallback_response": fix_result,
                        "all_attempts": all_attempts,
                        "total_attempts": attempt_num,
                        "message": "API connection failed. Fallback guidance provided.",
                        "ai_agents_used": ["gpt-4o" if not using_gemini else "gemini-1.5-pro"]
                    }
                
                # Check if Gemini was activated as fallback
                if fix_result.get("fallback_agent"):
                    using_gemini = True
                    print(f"[AGENT] 🔄 Now using Gemini AI as fallback")
                
                ai_agent = fix_result.get("ai_agent", "unknown")
                print(f"[AGENT] Fix generated by {ai_agent}. Confidence: {fix_result.get('confidence', 0):.0%}")
                print(f"[AGENT] Verifying fix in sandbox...")
                
                # Verify the fix
                verification = await self.verify_fix(
                    fixed_code=fix_result['fixed_code'],
                    original_error=error_message
                )
                
                # Record this attempt
                attempt_record = {
                    "attempt_number": attempt_num,
                    "ai_agent": ai_agent,
                    "fix_result": fix_result,
                    "verification": verification,
                    "fixed_code": fix_result['fixed_code'],
                    "verification_result": "PASSED" if verification['verified'] else "FAILED",
                    "new_error": verification.get('new_error'),
                    "timestamp": datetime.now().isoformat()
                }
                all_attempts.append(attempt_record)
                
                # If fix is verified, we're done!
                if verification['verified']:
                    print(f"[AGENT] ✅ Fix verified successfully on attempt {attempt_num} using {ai_agent}!")
                    return {
                        "success": True,
                        "final_fix": fix_result,
                        "final_verification": verification,
                        "all_attempts": all_attempts,
                        "total_attempts": attempt_num,
                        "message": f"Bug fixed successfully on attempt {attempt_num} using {ai_agent}",
                        "ai_agents_used": list(set([a["ai_agent"] for a in all_attempts])),
                        "gemini_used": using_gemini
                    }
                else:
                    print(f"[AGENT] ❌ Fix failed verification. Error: {verification.get('new_error', 'Unknown')[:100]}...")
                    
                    # If this was the last attempt, trigger final fallback
                    if attempt_num == max_attempts:
                        print(f"[AGENT] Maximum retry attempts ({max_attempts}) reached.")
                        
                        # Try switching to Gemini if we haven't already and it's available
                        if not using_gemini and self.gemini_agent and attempt_num < max_attempts:
                            print("[AGENT] 🔄 Attempting Gemini AI fallback for remaining attempts...")
                            using_gemini = True
                            continue
                        
                        # All AI attempts exhausted - use manual fallback if enabled
                        if self.fallback_handler.is_enabled():
                            print("[AGENT] Generating graceful fallback response...")
                            fallback_response = self.fallback_handler.generate_fallback_response(
                                error_message=error_message,
                                code_snippet=code_snippet,
                                bug_id=self.bug_id,
                                total_attempts=attempt_num,
                                all_attempts=all_attempts
                            )
                            
                            return {
                                "success": False,
                                "final_fix": None,
                                "fallback_response": fallback_response,
                                "all_attempts": all_attempts,
                                "total_attempts": attempt_num,
                                "message": f"Failed to fix bug after {max_attempts} attempts with multiple AI agents. Fallback guidance provided.",
                                "ai_agents_used": list(set([a["ai_agent"] for a in all_attempts])),
                                "gemini_used": using_gemini
                            }
                        else:
                            print("[AGENT] Fallback disabled. Returning failure.")
                            return {
                                "success": False,
                                "final_fix": None,
                                "all_attempts": all_attempts,
                                "total_attempts": attempt_num,
                                "message": f"Failed to fix bug after {max_attempts} attempts. Manual review needed.",
                                "last_error": verification.get('new_error'),
                                "ai_agents_used": list(set([a["ai_agent"] for a in all_attempts])),
                                "gemini_used": using_gemini
                            }
                    else:
                        # Switch to Gemini if GPT-4o fails and we haven't tried Gemini yet
                        if not using_gemini and self.gemini_agent:
                            print(f"[AGENT] 🔄 Switching to Gemini AI for next attempt...")
                            using_gemini = True
                        else:
                            print(f"[AGENT] Retrying with improved approach...")
            
            except Exception as e:
                print(f"[AGENT] ❌ Unexpected error on attempt {attempt_num}: {str(e)}")
                
                # Try Gemini if not already using it
                if not using_gemini and self.gemini_agent:
                    print(f"[AGENT] 🔄 Switching to Gemini AI after error...")
                    using_gemini = True
                    continue
                
                # If fallback is enabled, return it
                if self.fallback_handler.is_enabled():
                    fallback_response = self.fallback_handler.generate_api_failure_response(
                        error_message=error_message,
                        bug_id=self.bug_id,
                        api_error=str(e)
                    )
                    return {
                        "success": False,
                        "final_fix": None,
                        "fallback_response": fallback_response,
                        "all_attempts": all_attempts,
                        "total_attempts": attempt_num,
                        "message": f"Unexpected error: {str(e)}",
                        "ai_agents_used": list(set([a.get("ai_agent", "unknown") for a in all_attempts])),
                        "gemini_used": using_gemini
                    }
                else:
                    # Re-raise if fallback is disabled
                    raise
        
        # This shouldn't be reached, but just in case
        return {
            "success": False,
            "final_fix": None,
            "all_attempts": all_attempts,
            "total_attempts": len(all_attempts),
            "message": "Retry loop completed unexpectedly",
            "ai_agents_used": list(set([a.get("ai_agent", "unknown") for a in all_attempts])),
            "gemini_used": using_gemini
        }

    async def execute_full_workflow(
        self,
        error_message: str,
        code_snippet: str,
        file_path: str,
        repo_path: Optional[str] = None,
        use_retry: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the complete bug fixing workflow with real-time updates.
        
        Args:
            error_message: The error to fix
            code_snippet: Code containing the bug
            file_path: Path to the file
            repo_path: Optional repository path for git operations
            use_retry: Whether to use automatic retry logic (default: True)
            
        Yields:
            Status updates throughout the workflow
        """
        yield {
            "stage": "initialization",
            "message": f"Starting Bug Exorcist for {self.bug_id}",
            "timestamp": datetime.now().isoformat()
        }
        
        if use_retry:
            # Use the retry-enabled workflow with AI fallback
            gemini_status = "enabled" if self.gemini_agent else "not available"
            yield {
                "stage": "analysis",
                "message": f"Analyzing error with GPT-4o (retry logic enabled, Gemini fallback {gemini_status})...",
                "timestamp": datetime.now().isoformat()
            }
            
            retry_result = await self.analyze_and_fix_with_retry(
                error_message=error_message,
                code_snippet=code_snippet,
                file_path=file_path
            )
            
            # Yield updates for each attempt
            for attempt in retry_result['all_attempts']:
                ai_agent = attempt.get('ai_agent', 'unknown')
                yield {
                    "stage": f"attempt_{attempt['attempt_number']}",
                    "message": f"Attempt {attempt['attempt_number']} ({ai_agent}): {attempt['verification_result']}",
                    "data": attempt,
                    "timestamp": attempt['timestamp']
                }
            
            if retry_result['success']:
                ai_agents_used = ", ".join(retry_result.get('ai_agents_used', ['unknown']))
                yield {
                    "stage": "complete",
                    "message": f"Bug successfully exorcised on attempt {retry_result['total_attempts']} using {ai_agents_used}!",
                    "data": {
                        "fix": retry_result['final_fix'],
                        "verification": retry_result['final_verification'],
                        "total_attempts": retry_result['total_attempts'],
                        "ai_agents_used": retry_result.get('ai_agents_used', []),
                        "gemini_used": retry_result.get('gemini_used', False)
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Check if fallback was provided
                if 'fallback_response' in retry_result:
                    yield {
                        "stage": "fallback",
                        "message": f"AI analysis failed. Providing manual debugging guidance.",
                        "data": retry_result['fallback_response'],
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    yield {
                        "stage": "failed",
                        "message": f"Failed after {retry_result['total_attempts']} attempts. {retry_result['message']}",
                        "data": retry_result,
                        "timestamp": datetime.now().isoformat()
                    }
        else:
            # Original single-attempt workflow
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


# Convenience function for retry-enabled fixing
async def fix_with_retry(
    error: str,
    code: str,
    api_key: Optional[str] = None,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """
    Fix with automatic retry logic, AI fallback, and graceful manual guidance.
    
    Args:
        error: Error message
        code: Code with the bug
        api_key: Optional OpenAI API key
        max_attempts: Maximum retry attempts (default: 3)
        
    Returns:
        Dictionary with fix results, retry information, AI agents used, and fallback guidance if needed
    """
    agent = BugExorcistAgent(bug_id="retry-fix", openai_api_key=api_key)
    result = await agent.analyze_and_fix_with_retry(
        error_message=error,
        code_snippet=code,
        max_attempts=max_attempts
    )
    return result