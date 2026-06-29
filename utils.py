"""Utility helpers for the deep research workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from deep_research_from_scratch.prompts import summarize_webpage_prompt

try:
    from tavily import TavilyClient
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tavily-python"])
    from tavily import TavilyClient


class Summary(BaseModel):
    """Structured summary used for webpage content compression."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


def get_today_str() -> str:
    """Return today's date in a human-readable format."""

    now = datetime.now()
    return now.strftime(f"%a %b {now.day}, %Y")


def get_current_dir() -> Path:
    """Get the current directory of the module or notebook."""

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


summarization_model = init_chat_model(
    model="gemma-4-31b-it",
    model_provider="google_genai",
    temperature=0,
)

tavily_client = TavilyClient()


def tavily_client_search(
    query: str,
    max_results: int = 3,
    include_raw_content: bool = True,
    topic: Literal["general", "news", "finance"] = "general",
) -> dict:
    """Execute a Tavily search request and return the raw response."""

    try:
        return tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed for query {query!r}: {exc}") from exc


def tavily_search_multiple(
    search_queries: list[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> list[dict]:
    """Run multiple Tavily searches and collect their responses."""

    search_docs = []
    for query in search_queries:
        result = tavily_client_search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        search_docs.append(result)
    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage text into a structured block."""

    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary = structured_model.invoke(
            [
                HumanMessage(
                    content=summarize_webpage_prompt.format(
                        webpage_content=webpage_content,
                        date=get_today_str(),
                    )
                )
            ]
        )

        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except Exception as exc:
        print(f"Failed to summarize webpage: {exc}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content


def deduplicate_search_results(search_results: list[dict]) -> dict:
    """Deduplicate Tavily results by URL."""

    unique_results = {}
    for response in search_results:
        for result in response.get("results", []):
            url = result.get("url")
            if url and url not in unique_results:
                unique_results[url] = result
    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """Summarize search results where raw content is available."""

    summarized_results = {}

    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content = result.get("content", "")
        else:
            content = summarize_webpage_content(result["raw_content"])

        summarized_results[url] = {
            "title": result.get("title", "Untitled result"),
            "content": content,
        }

    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """Format processed search results for display or model consumption."""

    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for index, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {index}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """

    search_results = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        # Keep calls responsive by using Tavily's extracted content directly.
        include_raw_content=False,
    )

    unique_results = deduplicate_search_results(search_results)
    summarized_results = process_search_results(unique_results)
    return format_search_output(summarized_results)


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making."""

    return f"Reflection recorded: {reflection}"
