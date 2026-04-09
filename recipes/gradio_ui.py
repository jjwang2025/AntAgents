#!/usr/bin/env python
# coding=utf-8
#
# 重要说明：该脚本需要在 .env 里配置 DEEPSEEK_* 和 BOCHAAI_API_KEY。
# 对 GPT-5 / reasoning models 建议额外设置 OPENAI_API_MODE=auto。
#

import os
from dotenv import load_dotenv

from antagents import OpenAIServerModel, ToolCallingAgent, WebSearchTool
from antagents import GradioUI

load_dotenv()

api_mode = os.getenv("OPENAI_API_MODE", "auto")

model = OpenAIServerModel(
    model_id=os.getenv("DEEPSEEK_MODEL_ID"),
    api_base=os.getenv("DEEPSEEK_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_mode=api_mode,
)

agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=model,
    verbosity_level=1,
    planning_interval=3,
    name="HelloAgent",
    description="This is an example AntAgent.",
    step_callbacks=[],
    stream_outputs=True,
)

GradioUI(agent, file_upload_folder="./data").launch()
