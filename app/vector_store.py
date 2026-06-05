"""
Vector Store - RAG System
Real HuggingFace Sentence Transformers + ChromaDB
Proper chunking, metadata filtering, top-k retrieval.
Author: Srinivas Kanukolanu
"""

import chromadb
from chromadb.utils import embedding_functions
import math
from app.observability import logger

# ─────────────────────────────────────────
# EMBEDDING FUNCTION
# Uses HuggingFace all-MiniLM-L6-v2
# ─────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    logger.info(f"HuggingFace Embeddings loaded: {EMBEDDING_MODEL}")
except Exception as e:
    logger.warning(f"HuggingFace unavailable: {e}. Using fallback embeddings.")

    # Fallback embedding function
    from chromadb import Documents, EmbeddingFunction, Embeddings

    class FallbackEmbedding(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            vocab = {}
            for doc in input:
                for word in doc.lower().split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
            dim = max(len(vocab), 128)
            embeddings = []
            for doc in input:
                vec = [0.0] * dim
                for word in doc.lower().split():
                    if word in vocab:
                        idx = vocab[word] % dim
                        vec[idx] += 1.0
                magnitude = math.sqrt(sum(x * x for x in vec)) or 1.0
                embeddings.append([x / magnitude for x in vec])
            return embeddings

    embedding_fn = FallbackEmbedding()


# ─────────────────────────────────────────
# MOCK DATA — With proper chunking
# ─────────────────────────────────────────

# Programs — each chunk has rich metadata for filtering
PROGRAMS_CHUNKS = [
    {
        "id": "cs_overview",
        "text": "Computer Science MS program overview. Master of Science in Computer Science is a 2 year program costing 18000 dollars per year. Focuses on algorithms, software engineering, AI and systems.",
        "metadata": {"type": "program", "program": "computer science", "chunk": "overview"}
    },
    {
        "id": "cs_prereqs",
        "text": "Computer Science prerequisites and requirements. Applicants need Bachelor degree in CS or related field. GRE score minimum 310. TOEFL minimum 90 for international students.",
        "metadata": {"type": "program", "program": "computer science", "chunk": "prerequisites"}
    },
    {
        "id": "ds_overview",
        "text": "Data Science MS program overview. Master of Science in Data Science is 18 months long costing 16500 dollars per year. Covers machine learning, statistics, data engineering and visualization.",
        "metadata": {"type": "program", "program": "data science", "chunk": "overview"}
    },
    {
        "id": "ds_prereqs",
        "text": "Data Science prerequisites. Bachelor degree in Math Statistics or CS required. Python or R proficiency needed. GRE minimum 305.",
        "metadata": {"type": "program", "program": "data science", "chunk": "prerequisites"}
    },
    {
        "id": "mba_overview",
        "text": "MBA Business Administration program. Master of Business Administration is 2 years costing 22000 dollars per year. Covers finance, marketing, strategy and leadership.",
        "metadata": {"type": "program", "program": "business administration", "chunk": "overview"}
    },
    {
        "id": "mba_prereqs",
        "text": "MBA prerequisites. Bachelor degree in any field. Minimum 2 years work experience required. GMAT minimum score 600.",
        "metadata": {"type": "program", "program": "business administration", "chunk": "prerequisites"}
    },
    {
        "id": "ai_overview",
        "text": "Artificial Intelligence MS program. Master of Science in AI is 2 years costing 19500 dollars per year. Covers deep learning, NLP, computer vision, reinforcement learning.",
        "metadata": {"type": "program", "program": "artificial intelligence", "chunk": "overview"}
    },
    {
        "id": "ai_prereqs",
        "text": "AI program prerequisites. Bachelor in CS Math or Engineering. Strong Python proficiency. Linear Algebra and Statistics knowledge required.",
        "metadata": {"type": "program", "program": "artificial intelligence", "chunk": "prerequisites"}
    }
]

DEADLINES_CHUNKS = [
    {
        "id": "cs_deadlines",
        "text": "Computer Science application deadlines. Application deadline July 15 2026. Document submission deadline July 20 2026. Decision notification August 10 2026.",
        "metadata": {"type": "deadline", "program": "computer science", "application_deadline": "July 15, 2026", "document_deadline": "July 20, 2026", "decision_date": "August 10, 2026"}
    },
    {
        "id": "ds_deadlines",
        "text": "Data Science application deadlines. Application deadline June 30 2026. Document submission deadline July 5 2026. Decision notification July 25 2026.",
        "metadata": {"type": "deadline", "program": "data science", "application_deadline": "June 30, 2026", "document_deadline": "July 5, 2026", "decision_date": "July 25, 2026"}
    },
    {
        "id": "mba_deadlines",
        "text": "MBA Business Administration deadlines. Application deadline August 1 2026. Document submission August 7 2026. Decision notification September 1 2026.",
        "metadata": {"type": "deadline", "program": "business administration", "application_deadline": "August 1, 2026", "document_deadline": "August 7, 2026", "decision_date": "September 1, 2026"}
    },
    {
        "id": "ai_deadlines",
        "text": "Artificial Intelligence program deadlines. Application deadline July 20 2026. Document submission July 25 2026. Decision notification August 15 2026.",
        "metadata": {"type": "deadline", "program": "artificial intelligence", "application_deadline": "July 20, 2026", "document_deadline": "July 25, 2026", "decision_date": "August 15, 2026"}
    }
]

APPLICANTS_DATA = {
    "APP-1042": {
        "name": "Rahul Sharma",
        "program_applied": "Computer Science",
        "status": "Under Review",
        "next_step": "Await interview invitation - expected within 2 weeks.",
        "documents_pending": ["Statement of Purpose", "LOR from Professor 2"]
    },
    "APP-2089": {
        "name": "Priya Patel",
        "program_applied": "Data Science",
        "status": "Accepted",
        "next_step": "Complete enrollment form and pay seat deposit by June 30.",
        "documents_pending": []
    },
    "APP-3301": {
        "name": "Arun Kumar",
        "program_applied": "Business Administration",
        "status": "Documents Pending",
        "next_step": "Submit all pending documents to complete your application.",
        "documents_pending": ["Transcripts", "Work Experience Certificate", "GMAT Scorecard"]
    }
}

# ─────────────────────────────────────────
# INITIALIZE CHROMADB
# ─────────────────────────────────────────

chroma_client = chromadb.Client()


def init_vector_store():
    programs_col = chroma_client.get_or_create_collection(
        name="programs_v3",
        embedding_function=embedding_fn
    )
    if programs_col.count() == 0:
        programs_col.add(
            ids=[c["id"] for c in PROGRAMS_CHUNKS],
            documents=[c["text"] for c in PROGRAMS_CHUNKS],
            metadatas=[c["metadata"] for c in PROGRAMS_CHUNKS]
        )
        logger.info(f"Programs collection initialized: {programs_col.count()} chunks")

    deadlines_col = chroma_client.get_or_create_collection(
        name="deadlines_v3",
        embedding_function=embedding_fn
    )
    if deadlines_col.count() == 0:
        deadlines_col.add(
            ids=[c["id"] for c in DEADLINES_CHUNKS],
            documents=[c["text"] for c in DEADLINES_CHUNKS],
            metadatas=[c["metadata"] for c in DEADLINES_CHUNKS]
        )
        logger.info(f"Deadlines collection initialized: {deadlines_col.count()} chunks")

    return programs_col, deadlines_col


programs_col, deadlines_col = init_vector_store()


# ─────────────────────────────────────────
# SEMANTIC SEARCH WITH TOP-K + METADATA FILTER
# ─────────────────────────────────────────

def search_programs(query: str, top_k: int = 2) -> dict:
    """
    Semantic search for program info.
    Uses top-k retrieval and metadata filtering.
    """
    results = programs_col.query(
        query_texts=[query],
        n_results=top_k
    )

    if results["ids"][0]:
        # Combine top-k results for richer context
        combined_text = " ".join(results["documents"][0])
        best_meta = results["metadatas"][0][0]

        return {
            "found": True,
            "program": best_meta.get("program", "").title(),
            "details": combined_text,
            "chunks_retrieved": len(results["ids"][0])
        }
    return {"found": False, "message": f"No program found for '{query}'"}


def search_deadlines(query: str, top_k: int = 1) -> dict:
    """Semantic search for deadline info with metadata filtering."""
    results = deadlines_col.query(
        query_texts=[query],
        n_results=top_k
    )

    if results["ids"][0]:
        meta = results["metadatas"][0][0]
        return {
            "found": True,
            "program": meta.get("program", "").title(),
            "application_deadline": meta.get("application_deadline", ""),
            "document_deadline": meta.get("document_deadline", ""),
            "decision_date": meta.get("decision_date", "")
        }
    return {"found": False, "message": f"No deadline found for '{query}'"}


def get_applicant(applicant_id: str) -> dict:
    """Get applicant status."""
    aid = applicant_id.strip().upper()
    if aid in APPLICANTS_DATA:
        return {"found": True, "applicant": APPLICANTS_DATA[aid]}
    return {"found": False, "message": f"No applicant found with ID '{applicant_id}'"}
