import orjson
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import BaseCheckpointSaver, InMemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import BaseStore, InMemoryStore

from grug.ai_tools.dice_roller import roll_dice
from grug.ai_tools.image_generation import generate_ai_image
from grug.ai_tools.information_search import search_archives_of_nethys
from grug.ai_tools.reminders import set_reminder
from grug.settings import settings


def get_react_agent(
    checkpointer: BaseCheckpointSaver = InMemorySaver(),
    store: BaseStore = InMemoryStore(),
    debug: bool = False,
) -> CompiledGraph:
    return create_react_agent(
        model=ChatOpenAI(
            model_name=settings.ai_openai_model,
            temperature=0,
            max_tokens=None,
            max_retries=2,
            openai_api_key=settings.openai_api_key,
        ),
        tools=[roll_dice, search_archives_of_nethys, generate_ai_image, set_reminder],
        checkpointer=checkpointer,
        store=store,
        prompt=orjson.dumps(
            {
                "primary_instructions": [
                    f"your name is {settings.ai_name}.",
                    "You are an expert in Pathfinder 2E, and should leverage the archives of nethys for ALL pathfinder information.",
                    "You should respond conversationally, but remember to format your responses in markdown when providing information and details.",
                    "When the user is requesting a reminder or scheduling an event, ALWAYS use the tool, even if they just asked you to set a reminder.",
                ],
                "secondary_instructions": settings.ai_instructions if settings.ai_instructions else "",
            }
        ).decode(),
        debug=debug,
    )


# Create the ReAct agent for general and testing use.
graph = get_react_agent()

if __name__ == "__main__":
    # TODO: create functional tests for grug that check the output of the agent

    # TURN: 1
    results = graph.invoke(
        input={"messages": HumanMessage(content="Tell me about the wizard class.")},
        config={"configurable": {"thread_id": "default"}},
    )
    print(results["messages"][-1].content)

    # TURN: 2
    results = graph.invoke(
        input={"messages": HumanMessage(content="Tell me about yourself")},
        config={"configurable": {"thread_id": "default"}},
    )
    print(results["messages"][-1].content)
