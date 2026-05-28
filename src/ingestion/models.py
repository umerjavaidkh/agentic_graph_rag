from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class IngestionStatus(str, Enum):
    queued = "queued"
    validating = "validating"
    parsing = "parsing"
    building_structure = "building_structure"
    semantic_enrichment = "semantic_enrichment"
    exporting = "exporting"
    completed = "completed"
    failed = "failed"


class RelationConfig(BaseModel):
    field: str = Field(..., description="CSV column containing the target node id")
    target_label: str = Field(..., description="Label for the related target node")
    target_id_field: str = Field(..., description="Target node id field name")
    rel_type: str = Field(..., description="Relationship name, e.g. SUPPLIED_BY")
    direction: str = Field("out", description="Direction from source row to target row")

    @validator("direction")
    def validate_direction(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        return normalized


class StructuredCSVMapping(BaseModel):
    node_label: str = Field(..., description="Label for each CSV row node")
    id_field: str = Field(..., description="CSV column to use as the node id")
    properties: List[str] = Field(default_factory=list, description="Columns to store as node properties")
    relations: List[RelationConfig] = Field(default_factory=list, description="Foreign-key columns that become graph relationships")

    def validate_column_names(self, header: List[str]) -> None:
        missing = [self.id_field] + [prop for prop in self.properties if prop not in header]
        missing += [rel.field for rel in self.relations if rel.field not in header]
        missing += [rel.target_id_field for rel in self.relations if rel.target_id_field not in header]
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(set(missing))}")
