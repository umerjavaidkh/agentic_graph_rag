"""The MCP tools the router can dispatch to, one per retrieval axis.

Separated because the two axes share nothing but the access check: documents
answers from ingested PDFs, data from the business graph, hybrid runs both.
Keeping them in one 411-line module meant a change to either read as a change
to routing itself.

The registry stays here so adding a tool is one import and one entry, rather
than an edit inside whichever handler happened to be first.
"""
from .data import query_data
from .documents import search_documents
from .hybrid import query_hybrid
from .rbac import _rbac_check

MCP_HANDLERS = {
    "search_documents": search_documents,
    "query_data": query_data,
    "query_hybrid": query_hybrid,
}

MCP_TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search policy/compliance documents for procedures, protocols, sections, "
            "reporting rules, officer duties, whistleblowing, and organizational policy text."
        ),
        "fn": search_documents,
    },
    {
        "name": "query_data",
        "description": "Query structured Neo4j data (products, orders, customers, analytics, schema).",
        "fn": query_data,
    },
    {
        "name": "query_hybrid",
        "description": "Query both documents and structured data when both are required.",
        "fn": query_hybrid,
    },
]

__all__ = ["MCP_HANDLERS", "MCP_TOOLS", "query_data", "query_hybrid", "search_documents", "_rbac_check"]
