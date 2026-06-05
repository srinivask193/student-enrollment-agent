"""
Guardrails System
Input sanitization, prompt injection protection, output validation.
Author: Srinivas Kanukolanu
"""

import re
from app.observability import logger

# ─────────────────────────────────────────
# PROMPT INJECTION PATTERNS
# ─────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore (previous|all|system|above) (instructions?|prompts?|rules?)",
    r"forget (everything|all|previous|your instructions)",
    r"you are now",
    r"act as (a )?(different|new|evil|unrestricted)",
    r"jailbreak",
    r"bypass (safety|restrictions|rules|filters)",
    r"reveal (your|the) (system prompt|instructions|secrets)",
    r"pretend (you are|to be)",
    r"disregard (your|all) (instructions|rules)",
    r"override (your|the) (instructions|settings)",
    r"give me all (data|information|records)",
    r"show all (users|students|applicants|data)",
    r"dump (database|all data|records)",
]

# ─────────────────────────────────────────
# MAX INPUT LENGTH
# ─────────────────────────────────────────

MAX_INPUT_LENGTH = 500


# ─────────────────────────────────────────
# INPUT SANITIZATION
# ─────────────────────────────────────────

def sanitize_input(user_message: str, session_id: str = "") -> tuple[bool, str]:
    """
    Sanitize user input.
    Returns (is_safe, cleaned_message_or_error)
    """

    # Check length
    if len(user_message) > MAX_INPUT_LENGTH:
        logger.warning(f"INPUT_TOO_LONG | session={session_id[:8]} | length={len(user_message)}")
        return False, "Your message is too long. Please keep it under 500 characters."

    # Check for empty input
    if not user_message.strip():
        return False, "Please enter a message."

    # Check for prompt injection
    message_lower = user_message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message_lower):
            logger.warning(
                f"PROMPT_INJECTION_DETECTED | session={session_id[:8]} | "
                f"pattern={pattern} | message='{user_message[:50]}'"
            )
            return False, "I can only help with university enrollment queries. Please ask about programs, deadlines, or application status."

    # Remove special characters that could cause issues
    cleaned = re.sub(r'[<>{}|\[\]\\]', '', user_message)

    return True, cleaned.strip()


# ─────────────────────────────────────────
# OUTPUT VALIDATION
# ─────────────────────────────────────────

def validate_output(response: str) -> tuple[bool, str]:
    """
    Validate agent output before sending to user.
    Ensures response is appropriate.
    """

    if not response or not response.strip():
        return False, "I apologize, I could not generate a response. Please try again."

    # Check for sensitive data leakage patterns
    sensitive_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",   # SSN pattern
        r"\b\d{16}\b",                # Credit card
        r"password\s*[:=]\s*\S+",    # Password
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, response):
            logger.warning(f"SENSITIVE_DATA_IN_RESPONSE | pattern={pattern}")
            return False, "I cannot share that information. Please contact the admissions office directly."

    # Ensure response is not too long
    if len(response) > 2000:
        response = response[:2000] + "... [Response truncated. Please ask for more details.]"

    return True, response


# ─────────────────────────────────────────
# TOOL RESTRICTION LAYER
# ─────────────────────────────────────────

ALLOWED_TOOLS = ["get_program_info", "check_application_status", "get_deadlines"]

def validate_tool_call(tool_name: str) -> bool:
    """Ensure only allowed tools are called."""
    if tool_name not in ALLOWED_TOOLS:
        logger.warning(f"UNAUTHORIZED_TOOL_CALL | tool={tool_name}")
        return False
    return True
