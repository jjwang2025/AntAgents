#!/usr/bin/env python3

"""教科书式 ReAct 示例。

这个示例展示的是标准 ReAct（Reason + Act）循环，而不是 Plan-and-Execute。
ReAct 的特点是：模型在单个运行循环中，边思考边决定下一步动作，
并根据工具返回的 Observation 继续推进，直到可以给出最终答案。

整体流程如下：

    +-------------------+
    |     User Task     |
    +---------+---------+
              |
              v
    +-------------------+
    |  Agent / Model    |
    |  当前回合推理      |
    +---------+---------+
              |
              v
    +-------------------+
    | Action (Tool Call)|
    +---------+---------+
              |
              v
    +-------------------+
    | Observation       |
    | Tool Result       |
    +---------+---------+
              |
              v
    +-------------------+
    | Agent / Model     |
    | 基于观察继续推理   |
    +---------+---------+
              |
              v
    +-------------------+
    | final_answer      |
    +-------------------+

与 Plan-and-Execute 的区别：
1. ReAct 不先产出完整计划。
2. ReAct 每一步都根据最新 Observation 决定下一步动作。
3. 工具调用和推理是在同一个闭环中交替进行的。

这个脚本使用几个最小本地工具，避免示例依赖外部搜索服务，
让关注点保持在 ReAct 控制流本身。示例任务会触发多次工具调用，
从而更直观地展示 Action -> Observation -> Action -> Observation 的推进过程。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antagents import OpenAIServerModel, Tool, ToolCallingAgent
from antagents.agents import ActionOutput, ToolCall, ToolOutput
from antagents.memory import FinalAnswerStep


class DoubleNumberTool(Tool):
    """最小本地工具：把整数翻倍。

    这个工具足够简单，因此我们可以把注意力集中在：
    - 模型何时选择调用工具
    - 工具返回后如何继续推进
    - 何时结束并给出 final_answer
    """

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


class AddNumberTool(Tool):
    """最小本地工具：把两个整数相加。"""

    name = "add_number"
    description = "Adds two integers and returns the result."
    inputs = {
        "left": {
            "type": "integer",
            "description": "The first integer.",
        },
        "right": {
            "type": "integer",
            "description": "The second integer.",
        },
    }
    output_type = "integer"

    def forward(self, left: int, right: int) -> int:
        return left + right


class MultiplyNumberTool(Tool):
    """最小本地工具：把两个整数相乘。"""

    name = "multiply_number"
    description = "Multiplies two integers and returns the result."
    inputs = {
        "left": {
            "type": "integer",
            "description": "The first integer.",
        },
        "right": {
            "type": "integer",
            "description": "The second integer.",
        },
    }
    output_type = "integer"

    def forward(self, left: int, right: int) -> int:
        return left * right


def print_react_event(event, final_output: object | None) -> object | None:
    """把流式事件打印成教学友好的 ReAct 轨迹。

    这里打印的是框架暴露出来的关键节点：
    - ToolCall   -> Action
    - ToolOutput -> Observation
    - FinalAnswerStep / ActionOutput(final) -> 结束

    真实模型内部的完整思维链不会直接暴露；在工程实践里，我们通常只保留
    对调试最有价值的动作轨迹，而不是输出完整 chain-of-thought。
    """

    if isinstance(event, ToolCall):
        print(f"\nACTION: {event.name} {event.arguments}")
        return final_output

    if isinstance(event, ToolOutput):
        print(f"OBSERVATION: {event.output}")
        return final_output

    if isinstance(event, ActionOutput) and event.is_final_answer:
        final_output = event.output
        print(f"\nACTION_FINAL: {final_output}")
        return final_output

    if isinstance(event, FinalAnswerStep):
        final_output = event.output
        print(f"FINAL_ANSWER: {final_output}")
        return final_output

    return final_output


def main() -> None:
    """运行一个最小、可验证的 ReAct 闭环。"""

    load_dotenv(override=True)

    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID", "gpt-5"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_mode=os.getenv("OPENAI_API_MODE", "auto"),
    )

    agent = ToolCallingAgent(
        tools=[DoubleNumberTool(), AddNumberTool(), MultiplyNumberTool()],
        model=model,
        max_steps=8,
    )

    final_output = None
    task = "请按顺序调用工具：先把 21 翻倍，再加上 8，最后把结果乘以 3，然后只返回最终数字。"

    print("=== REACT EXAMPLE ===")
    print(f"TASK: {task}")

    for event in agent.run(task, stream=True):
        final_output = print_react_event(event, final_output)

    if str(final_output) != "150":
        raise SystemExit(f"Unexpected E2E result: {final_output!r}")

    print("\nRESULT:", final_output)


if __name__ == "__main__":
    main()
