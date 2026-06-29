"""State definitions for the deep research workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    researcher_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]


class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
