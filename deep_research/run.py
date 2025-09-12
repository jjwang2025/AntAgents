import argparse
import os
import threading

from dotenv import load_dotenv

from antagents import (
    GoogleSearchTool,
    WebSearchTool,
    GeminiServerModel,
    OpenAIServerModel,
    ToolCallingAgent,
    # TextWebView
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
    TextInspectorTool,
    visualizer,
)


append_answer_lock = threading.Lock()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent():
    model = OpenAIServerModel(
        model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        api_base=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

    text_limit = 30000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        WebSearchTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=8,
        verbosity_level=2,
        planning_interval=2,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web. If the original input includes an abbreviation, make sure your questions cover all situations including different possible full names.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = ToolCallingAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=6,
        verbosity_level=2,
        planning_interval=2,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent


def main():
    agent = create_agent()

    answer = agent.run("苏超哪个队会赢")
    
    print(f"Got this answer: {answer}")
    #agent.replay(detailed=True)

if __name__ == "__main__":
    load_dotenv()
    main()
