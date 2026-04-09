#!/usr/bin/env python3

"""GPT-5 Plan-and-Execute 示例。

这个示例实现的是更接近教科书定义的 Plan-and-Execute，而不是单个 ReAct agent
在执行过程中顺带生成计划。

整体流程如下：

    +-------------------+
    |   User Task       |
    +---------+---------+
              |
              v
    +-------------------+
    |  Planner (LLM)    |
    |  只负责产出计划   |
    +---------+---------+
              |
              v
    +-------------------+
    |  Structured Plan  |
    |  goal + steps[]   |
    +---------+---------+
              |
              v
    +-------------------+
    | Executor (Agent)  |
    | 逐步执行 steps[i] |
    +---------+---------+
              |
              v
    +-------------------+
    |  Step Results     |
    | 每步执行结果列表   |
    +---------+---------+
              |
              v
    +-------------------+
    | Synthesizer (LLM) |
    | 汇总最终答案      |
    +---------+---------+
              |
              v
    +-------------------+
    |  Final Answer     |
    +-------------------+

与 ReAct 的关键区别：
1. Planner 不执行工具，只负责把任务拆成计划。
2. Executor 不重新规划，只按既定步骤逐条执行。
3. Synthesizer 不调用工具，只基于执行结果做最终汇总。

因此，这个示例更适合展示 GPT-5 在“规划 -> 执行 -> 汇总”三阶段中的职责分离。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antagents import ChatMessage, MessageRole, OpenAIServerModel, Tool, ToolCallingAgent


class ArithmeticTool(Tool):
    """一个最小本地工具，用于让执行阶段真的通过工具完成计算。

    这里故意不接外部搜索或浏览器工具，目的是把示例聚焦在
    Plan-and-Execute 的控制流上，而不是环境依赖上。
    """

    name = "arithmetic"
    description = (
        "Evaluates a basic arithmetic expression and returns the numeric result. "
        "Use it for calculations instead of mental math."
    )
    inputs = {
        "expression": {
            "type": "string",
            "description": "A basic arithmetic expression using numbers and + - * / parentheses.",
        }
    }
    output_type = "string"

    def forward(self, expression: str) -> str:
        """执行一个受限的四则运算表达式。

        这里使用字符白名单 + 空 builtins 的方式，把工具能力限制在
        demo 所需的简单算术范围内，避免示例被无关能力干扰。
        """

        allowed_chars = set("0123456789+-*/(). ")
        if any(char not in allowed_chars for char in expression):
            raise ValueError("Expression contains unsupported characters.")
        # This demo tool intentionally supports only simple arithmetic expressions.
        return str(eval(expression, {"__builtins__": {}}, {}))


PLANNER_PROMPT = """你是一个 Plan-and-Execute 系统中的 Planner。

你的职责只有一件事：先产出一份可执行计划，不要执行计划，不要调用工具，不要直接回答任务。

请返回严格 JSON，对象格式如下：
{
  "goal": "对任务的简短重述",
  "steps": [
    "步骤1",
    "步骤2"
  ]
}

要求：
1. 只输出 JSON，不要输出任何额外说明。
2. steps 必须是可以执行的高层步骤列表。
3. 每一步都应尽量简洁、明确、可验证。
4. 如果任务涉及计算，请把“使用 arithmetic 工具完成计算”写进合适的步骤中。
"""


def build_model() -> OpenAIServerModel:
    """构造统一复用的 GPT-5/OpenAI-compatible 模型实例。

    示例继续沿用本仓库现有的 `DEEPSEEK_*` 环境变量命名，
    但当模型 ID 是 GPT-5 / reasoning model 时，`api_mode=auto`
    会自动切到 `responses` 协议。
    """

    return OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID", "gpt-5"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_mode=os.getenv("OPENAI_API_MODE", "auto"),
    )


def create_planner_messages(task: str) -> list[ChatMessage]:
    """为 Planner 构造消息。

    这里不用 ToolCallingAgent，而是直接用普通模型调用，原因是
    Planner 的职责是“生成结构化计划”，而不是“决定要不要调用工具”。
    """

    return [
        ChatMessage(role=MessageRole.SYSTEM, content=PLANNER_PROMPT),
        ChatMessage(role=MessageRole.USER, content=f"任务：{task}"),
    ]


def generate_plan(model: OpenAIServerModel, task: str) -> dict:
    """第一阶段：让 Planner 生成结构化计划。

    返回值必须是一个 JSON 对象，至少包含：
    - goal: 对任务的简短重述
    - steps: 可执行的步骤列表

    这里显式要求 `json_object`，是为了把 Planner 输出稳定约束为
    可解析的计划，而不是自然语言散文。
    """

    response = model.generate(messages=create_planner_messages(task), response_format={"type": "json_object"})
    if not isinstance(response.content, str):
        raise ValueError("Planner did not return text content.")

    plan = json.loads(response.content)
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ValueError(f"Invalid planner output: {response.content}")
    if not all(isinstance(step, str) and step.strip() for step in plan["steps"]):
        raise ValueError(f"Planner returned non-string or empty steps: {response.content}")
    return plan


def execute_step(task: str, plan: dict[str, Any], previous_results: list[dict[str, str]], step_index: int, model: OpenAIServerModel) -> str:
    """第二阶段：只执行单个计划步骤。

    这是 Plan-and-Execute 与 ReAct 的核心分界点之一：
    - Executor 只接收“当前步骤”
    - 不负责重新规划
    - 不负责提前执行后续步骤

    `previous_results` 会作为上下文传入，使当前步骤可以引用前面步骤的结果，
    但不会让模型忘记“当前只做一步”的边界。
    """

    agent = ToolCallingAgent(
        tools=[ArithmeticTool()],
        model=model,
        max_steps=4,
    )

    completed_steps = "\n".join(
        f"- 步骤 {item['step_index']}: {item['step']} -> {item['result']}" for item in previous_results
    ) or "- 暂无"
    current_step = plan["steps"][step_index]
    numbered_steps = "\n".join(f"{index}. {step}" for index, step in enumerate(plan["steps"], start=1))
    execution_task = (
        f"原始任务：{task}\n\n"
        f"总体目标：{plan.get('goal', task)}\n\n"
        f"完整计划：\n{numbered_steps}\n\n"
        f"已完成步骤结果：\n{completed_steps}\n\n"
        f"当前只执行这一步，不要重规划，也不要提前执行后续步骤：\n{step_index + 1}. {current_step}\n\n"
        "执行要求：\n"
        "1. 只完成当前步骤。\n"
        "2. 需要计算时调用 arithmetic 工具。\n"
        "3. 使用 final_answer 返回当前步骤的执行结果。"
    )
    return agent.run(execution_task)


def execute_plan(task: str, plan: dict[str, Any], model: OpenAIServerModel) -> list[dict[str, str]]:
    """顺序执行整份计划，并收集每一步的结构化结果。

    输出的 `results` 会成为第三阶段 Synthesizer 的输入。
    这一步故意把“计划”和“执行结果”分离存储，便于后续：
    - 调试
    - 人工审阅
    - 失败后重规划
    - 结果可追踪
    """

    results: list[dict[str, str]] = []
    for step_index, step in enumerate(plan["steps"]):
        print(f"\n=== EXECUTE STEP {step_index + 1} ===")
        print(step)
        result = str(execute_step(task, plan, results, step_index, model))
        print(f"STEP_RESULT: {result}")
        results.append(
            {
                "step_index": str(step_index + 1),
                "step": step,
                "result": result,
            }
        )
    return results


def synthesize_final_answer(task: str, plan: dict[str, Any], step_results: list[dict[str, str]], model: OpenAIServerModel) -> str:
    """第三阶段：基于计划和执行结果汇总最终答案。

    注意这里不再调用工具，也不再允许重新规划。
    这个阶段的职责是把“计划 + 各步结果”压缩成用户真正想看的最终答复。
    """

    synthesis_prompt = (
        "你是一个 Plan-and-Execute 系统中的 Synthesizer。"
        "请基于给定任务、计划和每一步的执行结果，生成最终答案。"
        "不要重新规划，不要调用工具，不要编造不存在的结果。"
        "请用中文简短回答，并明确说明最终使用了什么工具。"
    )
    results_text = "\n".join(
        f"步骤 {item['step_index']}: {item['step']}\n结果: {item['result']}" for item in step_results
    )
    response = model.generate(
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=synthesis_prompt),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    f"原始任务：{task}\n\n"
                    f"计划：{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
                    f"执行结果：\n{results_text}"
                ),
            ),
        ]
    )
    if not isinstance(response.content, str):
        raise ValueError("Synthesizer did not return text content.")
    return response.content


def main() -> None:
    """运行完整的 Planner -> Executor -> Synthesizer 示例。"""

    load_dotenv(override=True)

    task = (
        "先制定计划，再执行。"
        "请计算 ((18 + 24) * 3 - 12) / 6 的结果，"
        "并说明你使用了什么工具。最终答案请用中文简短回答。"
    )

    model = build_model()

    # Phase 1: planning
    plan = generate_plan(model, task)

    print("=== PLAN ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    # Phase 2: step-by-step execution
    step_results = execute_plan(task, plan, model)

    # Phase 3: final synthesis
    result = synthesize_final_answer(task, plan, step_results, model)

    print("\n=== EXECUTION RESULT ===")
    print(result)


if __name__ == "__main__":
    main()
