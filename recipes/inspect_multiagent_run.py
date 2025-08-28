import os
from dotenv import load_dotenv

from antagents import (
    OpenAIServerModel,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)


load_dotenv()

model = OpenAIServerModel(
    model_id=os.getenv("DEEPSEEK_MODEL_ID"),
    api_base=os.getenv("DEEPSEEK_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    return_full_result=True,
)

manager_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    return_full_result=True,
)
run_result = manager_agent.run(
    "如果中国保持2024年的经济增长率，GDP需要多少年能够翻倍？"
)
print("Here is the token usage for the manager agent", run_result.token_usage)
print("Here are the timing informations for the manager agent:", run_result.timing)
