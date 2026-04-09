#!/usr/bin/env python
# coding=utf-8
#
# 重要说明：该脚本需要在 .env 里配置 DEEPSEEK_* 和 BOCHAAI_API_KEY
#

import os
from dotenv import load_dotenv

from antagents import OpenAIServerModel, ToolCallingAgent, WebSearchTool

load_dotenv(override=True)

api_mode = os.getenv("OPENAI_API_MODE", "auto")

model = OpenAIServerModel(
    model_id=os.getenv("DEEPSEEK_MODEL_ID"),
    api_base=os.getenv("DEEPSEEK_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_mode=api_mode,
)

try:
    tools = [WebSearchTool()]
    agent = ToolCallingAgent(tools=tools, model=model)
    agent.run("东北虎以最快的速度，需要多长时间能跑过郑州黄河大桥？")
    #agent.replay(detailed=True)
finally:
    pass
