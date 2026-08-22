"""What the settings screen is allowed to change, and what counts as valid.

Separate from the HTTP layer on purpose. This is the security boundary --
the allow-list of names a web page may write -- and it should be readable
and testable without FastAPI, a running app, or a request.

Deliberately an allow-list, not "every environment variable". Anything that
grants access or holds a credential stays out: an API key must never be
readable through a web page, and ALLOW_CYPHER_INGEST / ALLOW_DB_RESET are
safety switches whose whole value is that flipping them takes a deploy.

Every entry here is a performance or quality dial -- how many documents
ingest at once, which model does entity extraction, how much work Axis 2
does per document. Changing one badly makes ingestion slow or coarse; it
cannot expose data or destroy any.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class Setting(BaseModel):
    name: str
    group: str
    kind: str                      # int | float | bool | choice | text
    help: str
    default: str
    choices: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    #: Where the value is consumed. A worker-side setting needs the workers
    #: restarted, not just the API, and saying so is the difference between
    #: "it did nothing" and "I know why it did nothing yet".
    applies_to: str = "api+workers"


#: The allow-list. Order is display order.
SETTINGS: List[Setting] = [
    Setting(name="WORKER_REPLICAS", group="Ingestion throughput", kind="int",
            default="2", minimum=1, maximum=64, applies_to="compose",
            help="Documents ingested at once — rq runs one job per worker process. "
                 "Multiplies with NER concurrency below: replicas × that is the real "
                 "number of concurrent model calls, and what hits the provider's "
                 "per-minute limit. Changing this needs `docker compose up -d`, not "
                 "just a restart."),
    Setting(name="AXIS2_NER_CONCURRENCY", group="Ingestion throughput", kind="int",
            default="8", minimum=1, maximum=64, applies_to="workers",
            help="Parallel entity-extraction calls inside one document."),
    Setting(name="AXIS2_NER_BATCH_SIZE", group="Ingestion throughput", kind="int",
            default="3", minimum=1, maximum=64, applies_to="workers",
            help="Text units per extraction request. Higher means far fewer requests "
                 "for the same tokens — the right dial when a provider limits requests "
                 "rather than tokens. Too high risks a truncated response, which is "
                 "retried by splitting the batch."),
    Setting(name="AXIS2_LLM_PAIR_CONCURRENCY", group="Ingestion throughput", kind="int",
            default="6", minimum=1, maximum=32, applies_to="workers",
            help="Parallel relationship-grounding calls."),

    Setting(name="AXIS2_MODEL", group="Models", kind="text", default="gpt-4o-mini",
            applies_to="workers",
            help="Model used for entity extraction and relationship grounding. Rate "
                 "limits are per model, so switching models switches quota."),
    Setting(name="CHAT_MODEL", group="Models", kind="text", default="gpt-4o-mini",
            applies_to="api+workers",
            help="Model used for answering and summarising."),
    Setting(name="MODEL_PROVIDER", group="Models", kind="choice",
            choices=["openai", "anthropic", "gemini"], default="openai",
            applies_to="api+workers",
            help="Chat/synthesis provider. Embeddings always use OpenAI regardless."),

    Setting(name="AXIS2_MAX_LLM_PAIRS", group="Graph construction", kind="int",
            default="300", minimum=0, maximum=2000, applies_to="workers",
            help="Most relationship pairs grounded by a model per document. The single "
                 "largest consumer of requests on a large corpus."),
    Setting(name="AXIS2_MAX_SIMILARITY_EDGES_PER_NODE", group="Graph construction",
            kind="int", default="20", minimum=1, maximum=200, applies_to="workers",
            help="Cap on semantic edges per node. Higher connects more and costs more "
                 "to query."),
    Setting(name="AXIS2_GROUND_LLM_EDGES", group="Graph construction", kind="bool",
            default="true", applies_to="workers",
            help="Whether a model verifies relationship edges. Off is much cheaper and "
                 "noticeably less precise."),

    Setting(name="ENABLE_PAGE_VISION", group="Document parsing", kind="bool",
            default="false", applies_to="workers",
            help="Describe figures and charts with a vision model. Off means diagram "
                 "content never reaches the graph; on costs a call per qualifying page."),
    Setting(name="VISION_MAX_PAGES_PER_DOC", group="Document parsing", kind="int",
            default="25", minimum=0, maximum=500, applies_to="workers",
            help="Ceiling on vision-enriched pages per document."),
    Setting(name="PDF_PARSER_BACKEND", group="Document parsing", kind="choice",
            choices=["rtldoc", "light", "table-aware"], default="rtldoc",
            applies_to="workers",
            help="Which PDF parser runs. rtldoc is geometry-first with its own role "
                 "classification; light is the PyMuPDF fallback."),
]

_BY_NAME = {s.name: s for s in SETTINGS}
_TRUE = {"1", "true", "yes", "on"}


def _coerce(setting: Setting, raw: str) -> str:
    """Validate one value, returning it normalised. Raises on bad input."""
    value = (raw or "").strip()
    if setting.kind in ("int", "float"):
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{setting.name} must be a number, got {value!r}")
        if setting.minimum is not None and number < setting.minimum:
            raise ValueError(f"{setting.name} must be at least {setting.minimum:g}")
        if setting.maximum is not None and number > setting.maximum:
            raise ValueError(f"{setting.name} must be at most {setting.maximum:g}")
        return str(int(number)) if setting.kind == "int" else str(number)
    if setting.kind == "bool":
        return "true" if value.lower() in _TRUE else "false"
    if setting.kind == "choice":
        if value not in (setting.choices or []):
            raise ValueError(
                f"{setting.name} must be one of {', '.join(setting.choices or [])}"
            )
        return value
    if not value:
        raise ValueError(f"{setting.name} cannot be empty")
    return value

