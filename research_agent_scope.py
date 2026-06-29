"""User Clarification and Research Brief Generation workflow."""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)
from deep_research_from_scratch.state_scope import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
    AgentInputState,
)


def get_today_str() -> str:
    """Get current date in a human-readable format."""

    now = datetime.now()
    return now.strftime(f"%a %b {now.day}, %Y")


model = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
)


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """Determine whether the user's request needs clarification."""

    structured_output_model = model.with_structured_output(ClarifyWithUser)

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=clarify_with_user_instructions.format(
                    messages=get_buffer_string(messages=state["messages"]),
                    date=get_today_str(),
                )
            )
        ]
    )

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )

    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


def write_research_brief(state: AgentState):
    """Convert the conversation into a research brief."""

    structured_output_model = model.with_structured_output(ResearchQuestion)

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_research_topic_prompt.format(
                    messages=get_buffer_string(state.get("messages", [])),
                    date=get_today_str(),
                )
            )
        ]
    )

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")],
    }


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scope_research = deep_researcher_builder.compile()