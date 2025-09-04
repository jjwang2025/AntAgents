#!/usr/bin/env python
# coding=utf-8

import argparse
import os

from dotenv import load_dotenv

from antagents import Model, OpenAIServerModel, Tool, ToolCallingAgent
from antagents.default_tools import TOOL_MAPPING


leopard_prompt = "东北虎以最快的速度，需要多长时间能跑过郑州黄河大桥？"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a ToolCallingAgent with all specified parameters")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=leopard_prompt,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="OpenAIServerModel",
        help="The model type to use (e.g., OpenAIServerModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="deepseek-chat",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    return parser.parse_args()


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    if model_type == "OpenAIServerModel":
        return OpenAIServerModel(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            api_base=api_base or os.getenv("DEEPSEEK_URL"),
            model_id=model_id,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> None:
    env_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(env_path)

    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)

    available_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            available_tools.append(Tool.from_space(tool_name))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    print(f"Running agent with these tools: {tools}")
    agent = ToolCallingAgent(tools=available_tools, model=model)

    agent.run(prompt)


def main() -> None:
    args = parse_arguments()
    run_smolagent(
        args.prompt,
        args.tools,
        args.model_type,
        args.model_id,
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
