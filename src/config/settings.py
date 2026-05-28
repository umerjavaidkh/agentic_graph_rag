import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
AUTO_LOAD_TO_NEO4J = os.environ.get("AUTO_LOAD_TO_NEO4J", "true").lower() in ("1", "true", "yes")

# Store ingestion artifacts (CSV/Cypher) on local disk under output/.
# Default OFF for scalable deployments; enable for debugging/auditing.
STORE_INGESTION_ARTIFACTS = os.environ.get("STORE_INGESTION_ARTIFACTS", "false").lower() in ("1", "true", "yes")

# SECURITY: Allows uploading and executing arbitrary Cypher against Neo4j.
# Keep disabled in production unless you also add strong authentication.
ALLOW_CYPHER_INGEST = os.environ.get("ALLOW_CYPHER_INGEST", "false").lower() in ("1", "true", "yes")

# If true, cypher ingestion will skip GenAI embedding statements like genai.vector.encode(...)
# Useful for loading schema/data when Neo4j GenAI credentials are not configured.
CYPHER_INGEST_SKIP_GENAI = os.environ.get("CYPHER_INGEST_SKIP_GENAI", "false").lower() in ("1", "true", "yes")

# If true, delete uploaded temp files in tmp_ingest/ after jobs finish.
# Disable to keep raw inputs for debugging.
CLEANUP_TMP_INGEST = os.environ.get("CLEANUP_TMP_INGEST", "true").lower() in ("1", "true", "yes")
