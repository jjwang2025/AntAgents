#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antagents import OpenAIServerModel, Tool, ToolCallingAgent
from antagents.agents import ActionOutput, ToolCall, ToolOutput
from antagents.memory import FinalAnswerStep


class DoubleNumberTool(Tool):
    name = "double_number"
    description = "Returns double the provided integer. Use this tool before answering when asked to double a number."
    inputs = {
        "value": {
            "type": "integer",
            "description": "The integer to double.",
        }
    }
    output_type = "integer"

    def forward(self, value: int) -> int:
        return value * 2


def main() -> None:
    load_dotenv(override=True)

    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID", "gpt-5"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_mode=os.getenv("OPENAI_API_MODE", "auto"),
    )

    agent = ToolCallingAgent(
        tools=[DoubleNumberTool()],
        model=model,
        max_steps=6,
    )

    final_output = None
    for event in agent.run("请调用工具把 21 翻倍，然后只返回最终数字。", stream=True):
        if isinstance(event, ToolCall):
            print(f"\nTOOL_CALL: {event.name} {event.arguments}")
        elif isinstance(event, ToolOutput):
            print(f"\nTOOL_OUTPUT: {event.output}")
        elif isinstance(event, ActionOutput) and event.is_final_answer:
            final_output = event.output
            print(f"\nACTION_FINAL: {final_output}")
        elif isinstance(event, FinalAnswerStep):
            final_output = event.output
            print(f"\nFINAL_STEP: {final_output}")

    print("\nE2E result:", final_output)


if __name__ == "__main__":
    main()
