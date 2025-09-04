#!/usr/bin/env python
# coding=utf-8
#
# 重要说明：该脚本需要在 .env 里配置 DEEPSEEK_* 和 BOCHAAI_API_KEY
#

import os
from dotenv import load_dotenv

from antagents import OpenAIServerModel, ToolCallingAgent, WebSearchTool

load_dotenv()

model = OpenAIServerModel(
    model_id=os.getenv("DEEPSEEK_MODEL_ID"),
    api_base=os.getenv("DEEPSEEK_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

try:
    tools = [WebSearchTool()]
    agent = ToolCallingAgent(tools=tools, model=model)
    #agent.run("东北虎以最快的速度，需要多长时间能跑过郑州黄河大桥？")
    agent.run("北京、上海、天津哪个城市最热？")
    #agent.replay(detailed=True)
finally:
    pass
