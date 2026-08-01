"""Terminal Weather Agent — Week 1 lab."""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent import build_graph

SYSTEM_PROMPT = (
    "You are a weather assistant. When the user asks about weather in a city, "
    "use the get_weather tool to look it up before answering. Always base your "
    "answer on the tool's observation rather than guessing."
)


def main():
    load_dotenv()
    graph = build_graph()

    print("Weather Agent — type a city name (or 'quit' to exit)")
    while True:
        city = input("\nCity> ").strip()
        if city.lower() in {"quit", "exit"}:
            break
        if not city:
            continue

        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"What's the weather in {city}?"),
                ]
            }
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
