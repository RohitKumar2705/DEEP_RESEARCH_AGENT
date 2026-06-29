"""State definitions and schemas for research scoping workflow."""

import operator
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentInputState(MessagesState):
    """Input state containing only user conversation messages."""


class AgentState(MessagesState):
    """Main state shared across scoping workflow nodes."""

    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    final_report: str


class ClarifyWithUser(BaseModel):
    """Schema used to decide whether clarification is needed."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope.",
    )
    verification: str = Field(
        description=(
            "Message confirming research will start after the user provides"
            " necessary details."
        ),
    )


class ResearchQuestion(BaseModel):
    """Schema for the generated research brief."""

    research_brief: str = Field(
        description="A research question that will guide the research.",
    )
