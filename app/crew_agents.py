"""
CrewAI Multi-Agent System - Properly Integrated
Intent Agent + Retrieval Agent + Decision Agent + Action Agent
Author: Srinivas Kanukolanu
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as crewai_tool
from dotenv import load_dotenv
from app.vector_store import search_programs, search_deadlines, get_applicant
from app.observability import logger

load_dotenv()

LLM_MODEL =  "groq/llama-3.1-8b-instant"

# ─────────────────────────────────────────
# CREWAI TOOLS
# ─────────────────────────────────────────

@crewai_tool("Search Program Information")
def crew_get_program_info(program_name: str) -> str:
    """Search for university program details using RAG."""
    result = search_programs(program_name, top_k=2)
    if result["found"]:
        return f"Program: {result['program']}\nDetails: {result['details']}"
    return f"No program found for '{program_name}'."


@crewai_tool("Check Application Status")
def crew_check_status(applicant_id: str) -> str:
    """Check student application status by ID."""
    result = get_applicant(applicant_id)
    if result["found"]:
        a = result["applicant"]
        pending = ", ".join(a["documents_pending"]) if a["documents_pending"] else "None"
        return f"Name: {a['name']}\nProgram: {a['program_applied']}\nStatus: {a['status']}\nNext Step: {a['next_step']}\nPending: {pending}"
    return f"No applicant found with ID '{applicant_id}'."


@crewai_tool("Get Application Deadlines")
def crew_get_deadlines(program_name: str) -> str:
    """Get application deadlines for a program."""
    result = search_deadlines(program_name)
    if result["found"]:
        return f"Program: {result['program']}\nApplication: {result['application_deadline']}\nDocuments: {result['document_deadline']}\nDecision: {result['decision_date']}"
    return f"No deadlines found for '{program_name}'."


# ─────────────────────────────────────────
# 4 SPECIALIZED CREWAI AGENTS
# ─────────────────────────────────────────

def run_multi_agent_crew(user_query: str, session_context: str = "") -> str:
    """
    Run the full multi-agent crew.
    Agent 1: Intent Agent - classify query
    Agent 2: Retrieval Agent - RAG search
    Agent 3: Decision Agent - form response
    Agent 4: Action Agent - handle bookings/status
    """
    logger.info(f"CREW_START | query='{user_query[:50]}'")

    # Agent 1 — Intent Classification Agent
    intent_agent = Agent(
        role="Intent Classification Agent",
        goal="Understand what the student is asking and classify the intent",
        backstory="""You analyze student queries and classify them into categories:
        PROGRAM_INFO, DEADLINE_INFO, APPLICATION_STATUS, or OUT_OF_SCOPE.
        You are precise and accurate in understanding student intent.""",
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=True
    )

    # Agent 2 — Retrieval Agent (RAG)
    retrieval_agent = Agent(
        role="Information Retrieval Agent",
        goal="Retrieve accurate information from the knowledge base using RAG",
        backstory="""You are an expert at searching the university knowledge base.
        You always use tools to get accurate information and never make up facts.""",
        tools=[crew_get_program_info, crew_get_deadlines],
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False
    )

    # Agent 3 — Action Agent
    action_agent = Agent(
        role="Application Status Agent",
        goal="Check and report student application status and document requirements",
        backstory="""You handle all application-related queries.
        You check application status, pending documents, and next steps for students.""",
        tools=[crew_check_status],
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False
    )

    # Agent 4 — Decision Agent (Final Response)
    decision_agent = Agent(
        role="Response Decision Agent",
        goal="Formulate the final clear and helpful response for the student",
        backstory="""You take information from other agents and form a clear,
        warm, and helpful response. If something is outside scope, you say:
        'I'd recommend speaking with an enrollment counselor for that.'
        You never make up information.""",
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False
    )

    # Tasks
    intent_task = Task(
        description=f"""
        Classify this student query: "{user_query}"
        Context from previous conversation: {session_context}
        
        Classify as one of: PROGRAM_INFO, DEADLINE_INFO, APPLICATION_STATUS, OUT_OF_SCOPE
        Pass the classification and key entities to the next agent.
        """,
        expected_output="Classification and key entities extracted from the query.",
        agent=intent_agent
    )

    retrieval_task = Task(
        description=f"""
        Based on the intent classification, retrieve relevant information.
        Original query: "{user_query}"
        
        If PROGRAM_INFO or DEADLINE_INFO — use your tools to search.
        If APPLICATION_STATUS — pass to action agent.
        If OUT_OF_SCOPE — note that escalation is needed.
        """,
        expected_output="Retrieved information from knowledge base.",
        agent=retrieval_agent
    )

    action_task = Task(
        description=f"""
        If the query involves application status, check it now.
        Query: "{user_query}"
        Context: {session_context}
        
        Look for applicant IDs like APP-XXXX in the query or context.
        Use your tool to check status if an ID is found.
        """,
        expected_output="Application status information if applicable.",
        agent=action_agent
    )

    decision_task = Task(
        description=f"""
        Form the final response for the student based on all gathered information.
        Original query: "{user_query}"
        
        Combine all retrieved information into one clear, helpful response.
        Be warm and professional. Keep it concise.
        If out of scope: "I'd recommend speaking with an enrollment counselor for that. Would you like me to connect you?"
        """,
        expected_output="Final natural language response for the student.",
        agent=decision_agent
    )

    crew = Crew(
        agents=[intent_agent, retrieval_agent, action_agent, decision_agent],
        tasks=[intent_task, retrieval_task, action_task, decision_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    logger.info(f"CREW_COMPLETE | query='{user_query[:50]}'")
    return str(result)
