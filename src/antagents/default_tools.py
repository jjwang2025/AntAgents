#!/usr/bin/env python
# coding=utf-8

import json
import os
import requests

from typing import Dict, List
from dataclasses import dataclass
from typing import Any

from .tools import Tool

@dataclass  
class PreTool:
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        user_input = input(f"{question} => Type your answer here:")
        return user_input


class DuckDuckGoSearchTool(Tool):
    name = "duckduckgo_search"
    description = """Performs a DuckDuckGo web search using SerpAPI, then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform. If the original input includes an abbreviation, do not expand it into the full name."}}
    output_type = "string"

    def __init__(self, max_results=6, **kwargs):
        super().__init__()
        self.max_results = max_results
        # 从环境变量获取API密钥
        self.api_key = os.getenv("SERPAPI_API_KEY")  
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY environment variable not set!")

    def _call_serpapi(self, query: str) -> List[Dict]:
        """向SerpAPI的DuckDuckGo搜索端点发起请求"""
        params = {
            "q": query,
            "engine": "duckduckgo",
            "api_key": self.api_key,
            "kl": "us-en",  # 语言/区域设置
        }
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()  # 如果请求失败则抛出错误
        data = response.json()
        return data.get("organic_results", [])

    def forward(self, query: str) -> str:
        try:
            results = self._call_serpapi(query)[:self.max_results]
            if not results:
                raise Exception("No results found! Try a less restrictive/shorter query.")
            
            postprocessed_results = [
                f"[{result['title']}]({result['link']})\n{result['snippet']}"
                for result in results
            ]
            return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
        
        except Exception as e:
            return f"Error performing search: {str(e)}"


class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform. If the original input includes an abbreviation, do not expand it into the full name."}}
    output_type = "string"

    def __init__(self, max_results=6):
        super().__init__()
        self.max_results = max_results

    def forward(self, query: str) -> str:
        results = self.search_bochaai(query)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def parse_results(self, results: list) -> str:
        return "## Search Results\n\n" + "\n\n".join(
            [f"[{result['title']}]({result['link']})\n{result['description']}" for result in results]
        )

    def parse_bochaai_results(self, response_json: json) -> list:
        results = []
        # 从JSON结构中逐层提取webPages.value数组
        web_pages = response_json.get('data', {}).get('webPages', {})
        value_list = web_pages.get('value', [])
        
        for item in value_list:
            # 构建包含title、link、description的字典
            result_item = {
                'title': item.get('name', '').strip(),  # 标题对应name字段
                'link': item.get('url', '').strip(),    # 链接对应url字段
                'description': item.get('summary', '').strip()  # 描述对应summary字段
            }
            results.append(result_item)        
        return results

    def search_bochaai(self, query: str) -> list:
        import requests

        url = "https://api.bochaai.com/v1/web-search"
        headers = {"Authorization": os.getenv("BOCHAAI_API_KEY"),
                   "Content-Type": "application/json"}
        data = {"query": query,
                "summary": True,
                "freshness": "noLimit",
                "count": 6}

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # 检查请求是否成功
        return self.parse_bochaai_results(response.json())
    
    def _create_bochaai_parser(self):
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    # 如果所有部分都存在，则存储当前结果
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # 发送GET请求到URL，设置20秒超时
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # 对于错误状态码抛出异常

            # 手动设置正确的编码（新浪网通常使用utf-8或gb2312）
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding  # 自动检测编码

            # 将HTML内容转换为Markdown
            markdown_content = markdownify(response.text).strip()

            # 移除多余换行符
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class WikipediaSearchTool(Tool):
    """
    WikipediaSearchTool搜索维基百科并返回给定主题的摘要或全文，以及页面URL。

    属性:
        user_agent (str): 用于标识项目的自定义用户智能体字符串。根据维基百科API政策，这是必需的，详情请参阅：http://github.com/martin-majlis/Wikipedia-API/blob/master/README.rst
        language (str): 检索维基百科文章的语言。
                http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (str): 定义要获取的内容。可以是"summary"获取简短摘要，或"text"获取完整文章。
        extract_format (str): 定义输出格式。可以是`"WIKI"`或`"HTML"`。

    示例:
        >>> from antagents import ToolCallingAgent, OpenAIServerModel, WikipediaSearchTool
        >>> model = OpenAIServerModel(
        >>>     model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        >>>     api_base=os.getenv("DEEPSEEK_URL"),
        >>>     api_key=os.getenv("DEEPSEEK_API_KEY")
        >>> )
        >>> agent = ToolCallingAgent(
        >>>     tools=[
        >>>            WikipediaSearchTool(
        >>>                user_agent="MyResearchBot (myemail@example.com)",
        >>>                language="en",
        >>>                content_type="summary",  # 或 "text"
        >>>                extract_format="WIKI",
        >>>            )
        >>>        ],
        >>>     model=model,
        >>> )
        >>> agent.run("介绍一下钢铁是怎么炼成的")
    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia. If the original input includes an abbreviation, do not expand it into the full name.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "antagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type

        # 将字符串格式映射到wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class GoogleSearchTool(Tool):
    name = "google_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform. If the original input includes an abbreviation, do not expand it into the full name."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)

TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        WebSearchTool,
        DuckDuckGoSearchTool,
        GoogleSearchTool
    ]
}

__all__ = [
    "FinalAnswerTool",
    "UserInputTool",
    "WebSearchTool", # 调用博査BochaAI接口
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
]