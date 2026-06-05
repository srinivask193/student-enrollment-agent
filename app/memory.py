"""
Memory System - Short-term + Long-term Memory
Short-term: LangGraph state (current conversation)
Long-term: SQLite database (persistent across sessions)
Author: Srinivas Kanukolanu
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "enrollment_memory.db")

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────

def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # User profiles table - long-term memory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            session_id TEXT PRIMARY KEY,
            name TEXT,
            application_id TEXT,
            interested_course TEXT,
            email TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # Query history table - long-term memory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            agent_response TEXT,
            tool_used TEXT,
            timestamp TEXT
        )
    """)

    # Conversation summary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary (
            session_id TEXT PRIMARY KEY,
            summary TEXT,
            updated_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    return True


# ─────────────────────────────────────────
# SHORT-TERM MEMORY — LangGraph State
# ─────────────────────────────────────────

MAX_MESSAGES = 10  # Keep last 10 messages in state

def trim_messages(messages: list) -> list:
    """
    Message trimming strategy — keeps last MAX_MESSAGES.
    Prevents context window overflow in long conversations.
    """
    if len(messages) > MAX_MESSAGES:
        # Always keep system context, trim oldest messages
        return messages[-MAX_MESSAGES:]
    return messages


def summarize_if_needed(messages: list, session_id: str) -> str:
    """
    Memory summarization for long conversations.
    When conversation gets long, create a summary.
    """
    if len(messages) < 8:
        return ""

    # Extract key info from messages
    summary_parts = []
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            content = msg.content[:100]
            summary_parts.append(content)

    summary = f"Previous conversation summary: {'; '.join(summary_parts[:3])}"

    # Save summary to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO conversation_summary (session_id, summary, updated_at)
        VALUES (?, ?, ?)
    """, (session_id, summary, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return summary


# ─────────────────────────────────────────
# LONG-TERM MEMORY — SQLite Operations
# ─────────────────────────────────────────

def save_user_profile(session_id: str, profile_data: dict):
    """Save or update user profile in long-term memory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    existing = cursor.execute(
        "SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)
    ).fetchone()

    now = datetime.now().isoformat()

    if existing:
        # Update only non-null fields
        updates = []
        values = []
        for field in ["name", "application_id", "interested_course", "email"]:
            if profile_data.get(field):
                updates.append(f"{field} = ?")
                values.append(profile_data[field])
        if updates:
            updates.append("updated_at = ?")
            values.append(now)
            values.append(session_id)
            cursor.execute(
                f"UPDATE user_profiles SET {', '.join(updates)} WHERE session_id = ?",
                values
            )
    else:
        cursor.execute("""
            INSERT INTO user_profiles 
            (session_id, name, application_id, interested_course, email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            profile_data.get("name"),
            profile_data.get("application_id"),
            profile_data.get("interested_course"),
            profile_data.get("email"),
            now, now
        ))

    conn.commit()
    conn.close()


def get_user_profile(session_id: str) -> dict:
    """Retrieve user profile from long-term memory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT name, application_id, interested_course, email FROM user_profiles WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()

    if row:
        return {
            "name": row[0],
            "application_id": row[1],
            "interested_course": row[2],
            "email": row[3]
        }
    return {}


def save_query_history(session_id: str, user_message: str, agent_response: str, tool_used: str = ""):
    """Save query to history for long-term memory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_history (session_id, user_message, agent_response, tool_used, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, user_message, agent_response, tool_used, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_query_history(session_id: str, limit: int = 5) -> list:
    """Get recent query history for context."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT user_message, agent_response, tool_used, timestamp 
        FROM query_history 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (session_id, limit)).fetchall()
    conn.close()

    return [
        {"user": r[0], "agent": r[1], "tool": r[2], "time": r[3]}
        for r in reversed(rows)
    ]


def extract_profile_from_message(message: str) -> dict:
    """
    Extract user profile info from message automatically.
    If student mentions APP-1042, save it to their profile.
    """
    profile = {}

    # Extract application ID
    import re
    app_id_match = re.search(r'APP-\d+', message.upper())
    if app_id_match:
        profile["application_id"] = app_id_match.group()

    # Extract course interest
    courses = ["computer science", "data science", "business administration", "artificial intelligence", "mba"]
    for course in courses:
        if course in message.lower():
            profile["interested_course"] = course
            break

    return profile


# Initialize DB on import
init_db()
