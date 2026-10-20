"""Output schema validation.

The model must return one JSON object matching `AssistantResponse`. Validation
has three layers: (1) the text must contain parseable JSON, (2) the object must
match the Pydantic model in strict mode (no type coercion, no extra keys), and
(3) semantic rules that a type system cannot express, such as "every citation
must be a document that was actually retrieved".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

DocId = Annotated[str, StringConstraints(pattern=r"^KB-\d{3}$")]


class AssistantResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str = Field(min_length=1, max_length=1500)
    citations: list[DocId] = Field(max_length=5)
    confidence: Literal["high", "medium", "low"]
    topic: Literal["appointments", "billing", "services", "medication", "policies", "other"]
    needs_human_followup: bool


@dataclass
class SchemaResult:
    valid: bool
    data: AssistantResponse | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _extract_json(raw: str) -> tuple[str | None, list[str]]:
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        return fenced.group(1), ["response was wrapped in a markdown code fence"]
    if text.startswith("{"):
        return text, []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1], ["response contained text outside the JSON object"]
    return None, []


def validate_response(raw: str, retrieved_ids: set[str]) -> SchemaResult:
    payload, warnings = _extract_json(raw)
    if payload is None:
        return SchemaResult(False, errors=["no JSON object found in response"], warnings=warnings)
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        return SchemaResult(False, errors=[f"invalid JSON: {e.msg} (pos {e.pos})"], warnings=warnings)
    if not isinstance(obj, dict):
        return SchemaResult(False, errors=["top-level JSON value must be an object"], warnings=warnings)

    try:
        data = AssistantResponse.model_validate(obj)
    except ValidationError as e:
        errors = [f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors()]
        return SchemaResult(False, errors=errors, warnings=warnings)

    errors = []
    unknown = [c for c in data.citations if c not in retrieved_ids]
    if unknown:
        errors.append(f"citations reference documents that were not provided: {unknown}")
    if not data.citations and data.confidence != "low":
        errors.append("an answer without citations must have confidence 'low'")
    if len(set(data.citations)) != len(data.citations):
        warnings.append("duplicate citations")
    return SchemaResult(not errors, data if not errors else None, errors, warnings)
