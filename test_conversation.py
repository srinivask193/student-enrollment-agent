"""
Test Conversation - 5 Turn Demo
Uses REAL EnrollmentAgent v3.0
NO hardcoded responses. Real LangGraph + RAG + Memory + Guardrails.
Author: Srinivas Kanukolanu
"""

import uuid
from dotenv import load_dotenv
load_dotenv()

from app.agent import EnrollmentAgent

TURNS = [
    "Hi, what programs do you offer in computer science?",
    "What's the application deadline for that?",
    "I already applied. My ID is APP-1042. What's my status?",
    "Can I get a fee waiver?",
    "What documents do I still need to submit?"
]

if __name__ == "__main__":
    session_id = str(uuid.uuid4())

    print("=" * 65)
    print("  STUDENT ENROLLMENT ASSISTANT v3.0 - 5-TURN TEST")
    print("  LangGraph + CrewAI + RAG + Memory + Guardrails")
    print("  Author: Srinivas Kanukolanu")
    print("=" * 65)

    agent = EnrollmentAgent(session_id=session_id)

    for i, user_msg in enumerate(TURNS, 1):
        print("")
        print("-" * 65)
        print(f"Turn {i}")
        print("-" * 65)
        print(f"User     : {user_msg}")
        response = agent.chat(user_msg)
        print(f"Assistant: {response}")

    print("")
    print("-" * 65)
    print("Memory Summary:")
    print("-" * 65)
    import json
    summary = agent.get_memory_summary()
    print(f"Profile  : {summary['profile']}")
    print(f"Turns    : {summary['tracker']['total_turns']}")

    print("")
    print("=" * 65)
    print("  All 5 turns completed successfully.")
    print("=" * 65)
