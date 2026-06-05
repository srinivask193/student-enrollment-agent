"""
Observability System
Logs tool calls, latency, errors, and agent decisions.
Author: Srinivas Kanukolanu
"""

import logging
import time
import functools
from datetime import datetime

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("EnrollmentAgent")
tool_logger = logging.getLogger("ToolCalls")
perf_logger = logging.getLogger("Performance")
error_logger = logging.getLogger("Errors")


# ─────────────────────────────────────────
# TOOL CALL LOGGING
# ─────────────────────────────────────────

def log_tool_call(tool_name: str, args: dict, result: str, latency_ms: float):
    """Log every tool call with args, result, and latency."""
    tool_logger.info(
        f"TOOL_CALL | tool={tool_name} | args={args} | "
        f"latency={latency_ms:.1f}ms | result_length={len(result)}"
    )


def log_agent_decision(session_id: str, user_message: str, tool_selected: str):
    """Log agent reasoning decision."""
    logger.info(
        f"AGENT_DECISION | session={session_id[:8]} | "
        f"query='{user_message[:50]}' | tool_selected={tool_selected}"
    )


def log_response(session_id: str, latency_ms: float, response_length: int):
    """Log final response metrics."""
    perf_logger.info(
        f"RESPONSE | session={session_id[:8]} | "
        f"total_latency={latency_ms:.1f}ms | response_length={response_length}"
    )


def log_error(session_id: str, error: str, context: str = ""):
    """Log errors with context."""
    error_logger.error(
        f"ERROR | session={session_id[:8]} | error={error} | context={context}"
    )


def log_escalation(session_id: str, user_message: str):
    """Log graceful escalations."""
    logger.warning(
        f"ESCALATION | session={session_id[:8]} | "
        f"query='{user_message[:50]}' | reason=out_of_scope"
    )


# ─────────────────────────────────────────
# PERFORMANCE DECORATOR
# ─────────────────────────────────────────

def track_latency(func):
    """Decorator to track function execution latency."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            latency = (time.time() - start) * 1000
            perf_logger.info(f"LATENCY | func={func.__name__} | {latency:.1f}ms")
            return result
        except Exception as e:
            latency = (time.time() - start) * 1000
            error_logger.error(f"ERROR | func={func.__name__} | {latency:.1f}ms | {str(e)}")
            raise
    return wrapper


# ─────────────────────────────────────────
# SESSION TRACKER
# ─────────────────────────────────────────

class SessionTracker:
    """Track metrics per session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.turn_count = 0
        self.tool_calls = []
        self.errors = []

    def record_turn(self, user_msg: str, tool: str, latency_ms: float):
        self.turn_count += 1
        self.tool_calls.append({
            "turn": self.turn_count,
            "tool": tool,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        })
        log_agent_decision(self.session_id, user_msg, tool)

    def record_error(self, error: str):
        self.errors.append(error)
        log_error(self.session_id, error)

    def summary(self) -> dict:
        total_time = (time.time() - self.start_time) * 1000
        return {
            "session_id": self.session_id,
            "total_turns": self.turn_count,
            "total_time_ms": round(total_time, 1),
            "tool_calls": self.tool_calls,
            "errors": self.errors
        }
