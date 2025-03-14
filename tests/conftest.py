from typing import Callable

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from grug.settings import settings


class ResponseEvaluation(BaseModel):
    success: bool = Field(description="Whether the response meets the expectations.")


@pytest.fixture
def llm_response_checker() -> Callable[[str, str], bool]:
    """Create a function to check if the response meets the expectations."""

    def response_meets_expectations(response: str, expectations: str) -> bool:
        """Check if the response meets the expectations."""
        llm = ChatOpenAI(
            model_name="o3-mini",
            openai_api_key=settings.openai_api_key,
        )

        structured_llm = llm.with_structured_output(ResponseEvaluation)

        evaluation: ResponseEvaluation = structured_llm.invoke(
            input=[
                SystemMessage(
                    content=(
                        "Your job is to evaluate the response of an AI assistant determine if it meets users "
                        "expected outcome."
                    )
                ),
                HumanMessage(content=f"does the response '{response}' meet the expectations '{expectations}'?"),
            ]
        )

        return evaluation.success

    return response_meets_expectations
