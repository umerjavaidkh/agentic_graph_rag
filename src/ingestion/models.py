from __future__ import annotations

from enum import Enum
from typing import Optional


class IngestionStatus(str, Enum):
    queued = "queued"
    validating = "validating"
    parsing = "parsing"
    building_structure = "building_structure"
    semantic_enrichment = "semantic_enrichment"
    vision_enrichment = "vision_enrichment"
    exporting = "exporting"
    completed = "completed"
    failed = "failed"
