"""
Student Enrollment Assistant Agent v3.0
Features:
- LangGraph agent loop
- CrewAI multi-agent (properly integrated)
- Real HuggingFace RAG
- Short-term + Long-term memory
- Guardrails
- Observability
- Message trimming
- Streaming support
Author: Srinivas Kanukolanu
"""

import os
import time
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.tools import TOOLS
from app.memory import (
    trim_messages, summarize_if_needed,
    save_user_profile, get_user_profile,
    save_query_history, get_query_history,
    extract_profile_from_message
)
from app.guardrails import sanitize_input, validate_output
from app.observability import logger, log_response, log_escalation, SessionTracker

load_dotenv()

# ─────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

llm_with_tools = llm.bind_tools(TOOLS)

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful Student Enrollment Assistant for a university admissions office.
Help students with programs, deadlines, and application status.

RULES:
- ALWAYS use tools to answer. NEVER use your own knowledge for program/deadline info.
- Remember context — if student already gave applicant ID or program, reuse it.
- For out-of-scope questions: "I'd recommend speaking with an enrollment counselor. Would you like me to connect you?"
- Never make up information. If tool returns no result, say so honestly.
- Be warm, concise, and professional.
"""

# ─────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────

class AgentState(TypedDict):
    messages: list
    session_id: str


# ─────────────────────────────────────────
# LANGGRAPH NODES
# ─────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Main agent node — LLM reasons and decides tool to call."""
    # Trim messages to prevent context overflow
    trimmed = trim_messages(state["messages"])

    # Get long-term memory context
    profile = get_user_profile(state["session_id"])
    profile_context = ""
    if profile:
        parts = []
        if profile.get("application_id"):
            parts.append(f"Student's application ID: {profile['application_id']}")
        if profile.get("interested_course"):
            parts.append(f"Student is interested in: {profile['interested_course']}")
        if profile.get("name"):
            parts.append(f"Student's name: {profile['name']}")
        if parts:
            profile_context = "\n\nLong-term memory context:\n" + "\n".join(parts)

    system_content = SYSTEM_PROMPT + profile_context
    messages = [SystemMessage(content=system_content)] + trimmed

    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response], "session_id": state["session_id"]}


def should_continue(state: AgentState) -> str:
    """Router — tools or end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────

tool_node = ToolNode(TOOLS)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ─────────────────────────────────────────
# ENROLLMENT AGENT CLASS
# ─────────────────────────────────────────

class EnrollmentAgent:
    """
    Student Enrollment Assistant v3.0
    LangGraph + LangChain + CrewAI + RAG + Memory + Guardrails + Observability
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.conversation_history = []
        self.tracker = SessionTracker(session_id)

    def chat(self, user_message: str) -> str:
        """Process one turn with full guardrails, memory, and observability."""
        start_time = time.time()

        # 1. Input sanitization (Guardrails)
        is_safe, cleaned_or_error = sanitize_input(user_message, self.session_id)
        if not is_safe:
            return cleaned_or_error

        user_message = cleaned_or_error

        # 2. Extract and save profile info to long-term memory
        profile_data = extract_profile_from_message(user_message)
        if profile_data:
            save_user_profile(self.session_id, profile_data)

        # 3. Add to conversation history
        self.conversation_history.append(HumanMessage(content=user_message))

        # 4. Run LangGraph agent
        try:
            result = graph.invoke({
                "messages": self.conversation_history,
                "session_id": self.session_id
            })
            self.conversation_history = result["messages"]
            response = result["messages"][-1].content

        except Exception as e:
            self.tracker.record_error(str(e))
            logger.error(f"Agent error: {e}")
            response = "I encountered an issue. Please try again."

        # 5. Output validation (Guardrails)
        is_valid, response = validate_output(response)

        # 6. Save to long-term memory
        save_query_history(self.session_id, user_message, response)

        # 7. Log performance
        latency = (time.time() - start_time) * 1000
        log_response(self.session_id, latency, len(response))

        # 8. Summarize if conversation is long
        summarize_if_needed(self.conversation_history, self.session_id)

        return response

    def chat_with_crew(self, user_message: str) -> str:
        """
        Alternative: Use CrewAI multi-agent system instead of LangGraph.
        4-agent pipeline: Intent → Retrieval → Action → Decision
        """
        from app.crew_agents import run_multi_agent_crew

        is_safe, cleaned_or_error = sanitize_input(user_message, self.session_id)
        if not is_safe:
            return cleaned_or_error

        # Get context for crew
        history = get_query_history(self.session_id, limit=3)
        context = " | ".join([f"Q: {h['user'][:50]}" for h in history])

        response = run_multi_agent_crew(cleaned_or_error, context)
        save_query_history(self.session_id, user_message, response, "crew")
        return response

    def get_memory_summary(self) -> dict:
        """Get full memory summary for this session."""
        return {
            "session_id": self.session_id,
            "profile": get_user_profile(self.session_id),
            "recent_history": get_query_history(self.session_id),
            "tracker": self.tracker.summary()
        }

    def reset(self):
        """Reset short-term memory (long-term persists)."""
        self.conversation_history = []
        logger.info(f"SESSION_RESET | session={self.session_id}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    import uuid
    session_id = str(uuid.uuid4())

    print("=" * 60)
    print("  Student Enrollment Assistant v3.0")
    print("  LangGraph + CrewAI + RAG + Memory + Guardrails")
    print("=" * 60)
    print("Commands: 'quit' | 'reset' | 'crew' | 'memory'\n")

    agent = EnrollmentAgent(session_id=session_id)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Short-term memory reset. Long-term memory preserved.\n")
            continue
        if user_input.lower() == "memory":
            import json
            print(json.dumps(agent.get_memory_summary(), indent=2))
            continue
        if user_input.lower().startswith("crew "):
            # Use CrewAI multi-agent
            query = user_input[5:]
            response = agent.chat_with_crew(query)
        else:
            response = agent.chat(user_input)

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
