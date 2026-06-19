import json
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()


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

ROLE_MAP = {"human": "user", "ai": "assistant"}


def log_messages(messages: list) -> None:
    """Print every message the agent produced so you can inspect the flow."""
    print("\n--- Message trace ---")
    for index, message in enumerate(messages, start=1):
        label = message.__class__.__name__

        if isinstance(message, HumanMessage):
            print(f"{index}. [{label}] {message.content}")

        elif isinstance(message, SystemMessage):
            print(f"{index}. [{label}] {message.content}")

        elif isinstance(message, AIMessage):
            if message.content:
                print(f"{index}. [{label}] {message.content}")
            else:
                print(f"{index}. [{label}] (no text — tool call only)")

            for tool_call in message.tool_calls:
                print(
                    f"    -> tool call: {tool_call['name']}("
                    f"{json.dumps(tool_call['args'])})"
                )

        elif isinstance(message, ToolMessage):
            print(
                f"{index}. [{label}] {message.name} "
                f"(id: {message.tool_call_id}) -> {message.content}"
            )

        else:
            print(f"{index}. [{label}] {message.content}")

    print("--- End trace ---\n")


def _serialize_messages(messages: list) -> list[dict]:
    serialized = []
    for message in messages:
        role = ROLE_MAP.get(message.type, message.type)
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        serialized.append({"role": role, "content": content})
    return serialized


def get_response(messages: list[dict]) -> dict:
    """Run the agent and return a JSON-serializable answer."""
    result = agent.invoke({"messages": messages})
    updated_messages = result["messages"]

    log_messages(updated_messages)

    answer = ""
    for message in reversed(updated_messages):
        if isinstance(message, AIMessage) and message.content:
            answer = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            break

    return {
        "answer": answer,
        "messages": _serialize_messages(updated_messages),
    }


if __name__ == "__main__":
    messages = []
    print("Type your message (or 'quit' to exit).")
    print("Try: What's the weather in London and what time is it there?\n")

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
