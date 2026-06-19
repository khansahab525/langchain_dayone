import json
from datetime import datetime
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

load_dotenv()


class QuestionAnalysis(BaseModel):
    """Structured understanding of what the user is asking."""

    city: Optional[str] = Field(
        default=None,
        description="City mentioned in the question, if any",
    )
    intent: Literal["weather", "time", "general", "both"] = Field(
        description=(
            "weather = asking about weather, "
            "time = asking about time, "
            "both = asking for weather and time, "
            "general = anything else"
        ),
    )


class StructuredResponse(BaseModel):
    """Typed API response — not free-form text."""

    city: Optional[str]
    intent: str
    answer: str
    messages: list[dict]


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_current_time(city: str) -> str:
    """Get the current local time for a major city."""
    timezones = {
        "london": "Europe/London",
        "new york": "America/New_York",
        "san francisco": "America/Los_Angeles",
        "tokyo": "Asia/Tokyo",
        "paris": "Europe/Paris",
    }

    tz_name = timezones.get(city.lower())
    if tz_name is None:
        return (
            f"Unknown city: {city}. "
            "Try London, New York, San Francisco, Tokyo, or Paris."
        )

    now = datetime.now(ZoneInfo(tz_name))
    return f"The current time in {city.title()} is {now.strftime('%I:%M %p')}."


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant",
)

analysis_model = init_chat_model("openai:gpt-5-mini").with_structured_output(
    QuestionAnalysis
)

ROLE_MAP = {"human": "user", "ai": "assistant"}


def _serialize_messages(messages: list) -> list[dict]:
    serialized = []
    for message in messages:
        role = ROLE_MAP.get(message.type, message.type)
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        serialized.append({"role": role, "content": content})
    return serialized


def _latest_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message["content"]
    return ""


def analyze_question(user_input: str) -> QuestionAnalysis:
    """Use structured output to extract city and intent from the question."""
    return analysis_model.invoke(
        f"Analyze this user question and extract the city and intent:\n\n{user_input}"
    )


def get_response(messages: list[dict]) -> dict:
    """Run structured analysis + agent, return typed JSON."""
    user_input = _latest_user_message(messages)
    analysis = analyze_question(user_input)

    result = agent.invoke({"messages": messages})
    updated_messages = result["messages"]

    answer = ""
    for message in reversed(updated_messages):
        if isinstance(message, AIMessage) and message.content:
            answer = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            break

    response = StructuredResponse(
        city=analysis.city,
        intent=analysis.intent,
        answer=answer,
        messages=_serialize_messages(updated_messages),
    )
    return response.model_dump()


if __name__ == "__main__":
    messages = []
    print("Type your message (or 'quit' to exit).")
    print("Try: What time is it in Tokyo?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})
        response = get_response(messages)
        messages = response["messages"]
        print(json.dumps(response, indent=2))
        print()
