#!/usr/bin/env python
# coding=utf-8
#
# 重要说明：该脚本需要在 .env 里配置 DEEPSEEK_* 和 BOCHAAI_API_KEY。
# GPT-5 / OpenAI reasoning models 可使用 OPENAI_API_MODE=auto；DeepSeek 兼容端点通常继续走 chat_completions。
#

import os

from dotenv import load_dotenv

from antagents import GradioUI, OpenAIServerModel, ToolCallingAgent, VisitWebpageTool, WebSearchTool

load_dotenv()

api_mode = os.getenv("OPENAI_API_MODE", "auto")

model = OpenAIServerModel(
    model_id=os.getenv("DEEPSEEK_MODEL_ID"),
    api_base=os.getenv("DEEPSEEK_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_mode=api_mode,
)

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    verbosity_level=1,
    # Managed agent names are exposed as tool names, so they must stay ASCII-safe.
    name="scout_ant",
    description="侦察蚁：负责网页搜索与网页访问的检索子智能体。",
    return_full_result=True,
)

agent = ToolCallingAgent(
    tools=[],
    managed_agents=[search_agent],
    model=model,
    verbosity_level=1,
    planning_interval=3,
    name="antagents_console",
    description="多智能体编排演示：AntAgents 总控台负责规划与汇总，侦察蚁负责搜索与页面访问。",
    step_callbacks=[],
    stream_outputs=True,
)
agent.display_name = "AntAgents 总控台"

GradioUI(agent, file_upload_folder="./data").launch()
