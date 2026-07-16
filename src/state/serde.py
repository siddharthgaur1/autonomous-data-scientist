"""JSON round-tripping for RunState.

SQLite stores each transition as JSON, and a run must be replayable, so decoding
has to rebuild the Pydantic payloads rather than leaving them as bare dicts.
The field->model map below is the authority for that; a new typed field in
RunState needs an entry here or it round-trips as a plain dict.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .schema import (
    AgentMessage,
    Artifact,
    CandidateModel,
    EDAFindings,
    EngineeredFeature,
    EvalMetrics,
    RunState,
    RunStatus,
    TaskType,
    TokenUsage,
    Transformation,
    TuningResults,
)

# field -> model, for fields holding a single Pydantic payload
_SCALAR_MODELS: dict[str, type[BaseModel]] = {
    "eda_findings": EDAFindings,
    "tuning_results": TuningResults,
    "eval_metrics": EvalMetrics,
}

# field -> model, for fields holding a list of Pydantic payloads
_LIST_MODELS: dict[str, type[BaseModel]] = {
    "transformations": Transformation,
    "engineered_features": EngineeredFeature,
    "candidate_models": CandidateModel,
    "messages": AgentMessage,
    "artifacts": Artifact,
}

# field -> model, for fields holding a dict of str -> Pydantic payload
_DICT_MODELS: dict[str, type[BaseModel]] = {"token_usage": TokenUsage}

_ENUMS: dict[str, type] = {"task_type": TaskType, "status": RunStatus}


def _default(obj: Any) -> Any:
    """Fallback encoder for types json doesn't know."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def state_to_json(state: RunState) -> str:
    """Serialize a RunState to a JSON string."""
    return json.dumps(dict(state), default=_default)


def state_from_json(raw: str) -> RunState:
    """Rebuild a RunState, restoring Pydantic payloads and enums."""
    data: dict[str, Any] = json.loads(raw)

    for field, model in _SCALAR_MODELS.items():
        if data.get(field) is not None:
            data[field] = model.model_validate(data[field])

    for field, model in _LIST_MODELS.items():
        if data.get(field):
            data[field] = [model.model_validate(x) for x in data[field]]

    for field, model in _DICT_MODELS.items():
        if data.get(field):
            data[field] = {k: model.model_validate(v) for k, v in data[field].items()}

    for field, enum in _ENUMS.items():
        if data.get(field) is not None:
            data[field] = enum(data[field])

    return RunState(**data)  # type: ignore[typeddict-item]
