"""
LangChain Tools - 3 tools with observability
Author: Srinivas Kanukolanu
"""

import time
from langchain_core.tools import tool
from app.vector_store import search_programs, search_deadlines, get_applicant
from app.observability import log_tool_call


@tool
def get_program_info(program_name: str) -> str:
    """
    Get details about a university program including duration, tuition, and prerequisites.
    Use this when a student asks about any program offered by the university.
    """
    start = time.time()
    result = search_programs(program_name, top_k=2)
    latency = (time.time() - start) * 1000

    if result["found"]:
        response = (
            f"Program: {result['program']}\n"
            f"Details: {result['details']}"
        )
    else:
        response = f"No program found for '{program_name}'. Available: Computer Science, Data Science, Business Administration, Artificial Intelligence."

    log_tool_call("get_program_info", {"program_name": program_name}, response, latency)
    return response


@tool
def check_application_status(applicant_id: str) -> str:
    """
    Check the application status for a student using their applicant ID.
    Use this when a student asks about their application status or pending documents.
    """
    start = time.time()
    result = get_applicant(applicant_id)
    latency = (time.time() - start) * 1000

    if result["found"]:
        a = result["applicant"]
        pending = ", ".join(a["documents_pending"]) if a["documents_pending"] else "None - all submitted!"
        response = (
            f"Applicant: {a['name']}\n"
            f"Program: {a['program_applied']}\n"
            f"Status: {a['status']}\n"
            f"Next Step: {a['next_step']}\n"
            f"Documents Pending: {pending}"
        )
    else:
        response = f"No applicant found with ID '{applicant_id}'. Please verify your ID."

    log_tool_call("check_application_status", {"applicant_id": applicant_id}, response, latency)
    return response


@tool
def get_deadlines(program_name: str) -> str:
    """
    Get application deadlines, document submission deadlines, and decision dates.
    Use this when a student asks about deadlines or important dates.
    """
    start = time.time()
    result = search_deadlines(program_name, top_k=1)
    latency = (time.time() - start) * 1000

    if result["found"]:
        response = (
            f"Program: {result['program']}\n"
            f"Application Deadline: {result['application_deadline']}\n"
            f"Document Submission: {result['document_deadline']}\n"
            f"Decision Notification: {result['decision_date']}"
        )
    else:
        response = f"No deadline found for '{program_name}'."

    log_tool_call("get_deadlines", {"program_name": program_name}, response, latency)
    return response


TOOLS = [get_program_info, check_application_status, get_deadlines]
