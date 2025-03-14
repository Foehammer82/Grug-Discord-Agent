from langchain_core.messages import HumanMessage

from grug.ai_agent import graph


def test_basic_agent_usage(llm_response_checker):
    """
    Test the basic usage of the ReAct agent.
    This test checks if the agent can respond to a simple query about the wizard class.
    """

    results = graph.invoke(
        input={"messages": HumanMessage(content="Tell me about the wizard class.")},
        config={"configurable": {"thread_id": "default"}},
    )
    response = results["messages"][-1].content

    assert llm_response_checker(
        response, "gave information about the wizard class and provided links to sources."
    ).success
