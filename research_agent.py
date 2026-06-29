"""Deep research agent graph."""

from __future__ import annotations

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, filter_messages
from langgraph.graph import END, START, StateGraph

from deep_research_from_scratch.prompts import (
    compress_research_human_message,
    compress_research_system_prompt,
    research_agent_prompt,
)
from deep_research_from_scratch.state_research import ResearcherOutputState, ResearcherState
from deep_research_from_scratch.utils import get_today_str, tavily_search, think_tool


tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
)

model_with_tools = model.bind_tools(tools)
compress_model = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
)


def llm_call(state: ResearcherState):
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary."""

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [
        HumanMessage(content=compress_research_human_message)
    ]
    response = compress_model.invoke(messages)

    raw_notes = [
        str(message.content)
        for message in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Route to tool execution when the LLM requested tools."""

    # Prevent long-running loops when the model keeps requesting tools.
    tool_messages = filter_messages(state.get("researcher_messages", []), include_types=["tool"])
    if len(tool_messages) >= 3:
        return "compress_research"

    last_message = state["researcher_messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"


agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    },
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("compress_research", END)

researcher_agent = agent_builder.compile()
