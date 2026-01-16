"""
core/example_retry_usage.py - Examples demonstrating automatic retry logic

This file shows how the Bug Exorcist Agent automatically retries failed fixes.
"""

import asyncio
from core.agent import BugExorcistAgent, fix_with_retry


# Example 1: Simple bug that might need retry
async def example_division_by_zero():
    """
    Example where the first fix might not handle all edge cases,
    requiring a retry for a more robust solution.
    """
    print("=" * 60)
    print("EXAMPLE 1: Division by Zero with Edge Cases")
    print("=" * 60)
    
    error = """
ZeroDivisionError: division by zero
  File "calculator.py", line 5, in divide
    return numerator / denominator
"""
    
    code = """
def divide(numerator, denominator):
    return numerator / denominator

# Called with: divide(10, 0)
"""
    
    agent = BugExorcistAgent(bug_id="RETRY-001")
    
    result = await agent.analyze_and_fix_with_retry(
        error_message=error,
        code_snippet=code,
        max_attempts=3
    )
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    print(f"Total Attempts: {result['total_attempts']}")
    print(f"{'='*60}\n")
    
    # Show all attempts
    for i, attempt in enumerate(result['all_attempts'], 1):
        print(f"--- Attempt {i} ---")
        print(f"Status: {attempt['verification_result']}")
        if not attempt['verification']['verified']:
            print(f"Error: {attempt['new_error'][:100]}...")
        print()
    
    if result['success']:
        print("✅ Final Working Fix:")
        print(result['final_fix']['fixed_code'])
        print(f"\nExplanation: {result['final_fix']['explanation']}")


# Example 2: Type error requiring multiple attempts
async def example_type_mismatch():
    """
    Example where initial fix might not account for all type combinations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Type Mismatch with Multiple Cases")
    print("=" * 60)
    
    error = """
TypeError: unsupported operand type(s) for +: 'int' and 'str'
  File "processor.py", line 3, in add_values
    return a + b
"""
    
    code = """
def add_values(a, b):
    return a + b

# Called with: add_values(5, "10")
"""
    
    result = await fix_with_retry(
        error=error,
        code=code,
        max_attempts=3
    )
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
    print(f"Message: {result['message']}")
    print(f"{'='*60}\n")
    
    # Show progression
    print("📊 Attempt Progression:")
    for attempt in result['all_attempts']:
        status_icon = "✅" if attempt['verification']['verified'] else "❌"
        print(f"{status_icon} Attempt {attempt['attempt_number']}: {attempt['verification_result']}")


# Example 3: Complex bug requiring iterative refinement
async def example_list_index_error():
    """
    Example showing how the AI learns from failed attempts.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: List Index Error with Edge Cases")
    print("=" * 60)
    
    error = """
IndexError: list index out of range
  File "data_handler.py", line 2, in get_first_element
    return data[0]
"""
    
    code = """
def get_first_element(data):
    return data[0]

# Called with: get_first_element([])
"""
    
    agent = BugExorcistAgent(bug_id="RETRY-003")
    
    print("Starting retry workflow...\n")
    
    result = await agent.analyze_and_fix_with_retry(
        error_message=error,
        code_snippet=code,
        additional_context="This function is called from an API endpoint and must handle empty lists gracefully.",
        max_attempts=3
    )
    
    print(f"\n{'='*60}")
    print("DETAILED ATTEMPT ANALYSIS")
    print(f"{'='*60}\n")
    
    for attempt in result['all_attempts']:
        print(f"🔍 Attempt {attempt['attempt_number']}:")
        print(f"   Root Cause: {attempt['fix_result']['root_cause'][:80]}...")
        print(f"   Confidence: {attempt['fix_result']['confidence']:.0%}")
        print(f"   Verification: {attempt['verification_result']}")
        
        if attempt['fix_result'].get('retry_analysis'):
            print(f"   Retry Analysis: {attempt['fix_result']['retry_analysis'][:100]}...")
        
        print(f"   Code Preview:")
        code_preview = attempt['fixed_code'].split('\n')[:3]
        for line in code_preview:
            print(f"      {line}")
        print()
    
    if result['success']:
        print("=" * 60)
        print("✅ FINAL WORKING SOLUTION")
        print("=" * 60)
        print(result['final_fix']['fixed_code'])
        print(f"\n💡 Key Changes: {result['final_fix']['explanation']}")


# Example 4: Demonstrating max attempts limit
async def example_max_attempts_reached():
    """
    Example showing what happens when max attempts is reached.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Max Attempts Reached (Failure Case)")
    print("=" * 60)
    
    # Intentionally complex error that might exhaust retries
    error = """
RuntimeError: Cannot process data with conflicting constraints
  File "complex_processor.py", line 10, in process
    validate_and_transform(data)
"""
    
    code = """
def process(data):
    # Very complex logic with multiple edge cases
    if not data:
        raise ValueError("Empty data")
    
    for item in data:
        validate_and_transform(item)
    
    return result
"""
    
    agent = BugExorcistAgent(bug_id="RETRY-004")
    
    result = await agent.analyze_and_fix_with_retry(
        error_message=error,
        code_snippet=code,
        max_attempts=2  # Limit to 2 for demo
    )
    
    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED (Manual Intervention Needed)'}")
    print(f"Total Attempts: {result['total_attempts']}")
    print(f"Message: {result['message']}")
    print(f"{'='*60}\n")
    
    if not result['success']:
        print("⚠️  All automatic retry attempts exhausted.")
        print("📋 Summary of attempts:")
        for i, attempt in enumerate(result['all_attempts'], 1):
            print(f"   {i}. {attempt['verification_result']}")
        print("\n💡 Recommendation: Manual code review required")


# Example 5: Using retry with API endpoint
async def example_api_integration():
    """
    Example showing how to use retry logic via the API.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: API Integration with Retry")
    print("=" * 60)
    
    # This would be a POST request to /api/agent/fix-with-retry
    request_payload = {
        "error_message": "AttributeError: 'NoneType' object has no attribute 'strip'",
        "code_snippet": """
def clean_text(text):
    return text.strip().lower()
""",
        "file_path": "text_utils.py",
        "max_attempts": 3
    }
    
    print("API Request Payload:")
    print(f"  Endpoint: POST /api/agent/fix-with-retry")
    print(f"  Max Attempts: {request_payload['max_attempts']}")
    print(f"  Error: {request_payload['error_message']}")
    
    print("\nExpected Response Structure:")
    print("""
{
  "success": true,
  "final_fix": {
    "fixed_code": "...",
    "root_cause": "...",
    "explanation": "..."
  },
  "all_attempts": [
    {
      "attempt_number": 1,
      "verification_result": "FAILED",
      "new_error": "..."
    },
    {
      "attempt_number": 2,
      "verification_result": "PASSED"
    }
  ],
  "total_attempts": 2,
  "message": "Bug fixed successfully on attempt 2"
}
""")


async def main():
    """Run all examples"""
    print("\n🧟‍♂️ BUG EXORCIST - RETRY LOGIC EXAMPLES")
    print("=" * 60)
    print("Demonstrating automatic retry for failed fixes")
    print("=" * 60)
    
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning examples in demo mode...\n")
        
        # Run example 5 (API integration) which doesn't need actual API calls
        await example_api_integration()
        return
    
    # Run all examples
    examples = [
        ("Division by Zero", example_division_by_zero),
        ("Type Mismatch", example_type_mismatch),
        ("List Index Error", example_list_index_error),
        ("Max Attempts", example_max_attempts_reached),
        ("API Integration", example_api_integration)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(2)  # Brief pause between examples
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All retry logic examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())