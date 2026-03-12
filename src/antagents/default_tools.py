#!/usr/bin/env python
# coding=utf-8

import json
import os
import requests
import time
import xml.etree.ElementTree as ET

from typing import List, Any, Optional
from dataclasses import dataclass

from .utils import BASE_BUILTIN_MODULES
from .local_python_executor import (
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from .tools import Tool


@dataclass
class PreTool:
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    f"The code snippet to evaluate. All variables used in this snippet must be defined "
                    f"in this same snippet, else you will get an error. This code can only import the following "
                    f"python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
            )[0]  # The second element is boolean is_final_answer
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


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
    """Web search tool that performs searches using the DuckDuckGo search engine.

    Args:
        max_results (`int`, default `10`): Maximum number of search results to return.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.
        **kwargs: Additional keyword arguments for the `DDGS` client.

    Examples:
        ```python
        >>> from antagents import DuckDuckGoSearchTool
        >>> web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "duckduckgo_search"
    description = (
        "Performs a duckduckgo web search based on your query (think a Google search) "
        "then returns the top search results."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform."
        }
    }
    output_type = "string"

    def __init__(self, max_results: int = 10, rate_limit: float | None = 1.0, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        self._enforce_rate_limit()
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Performs a web search for a query and returns a string of the top search results formatted "
        "as markdown with titles, links, and descriptions."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform. If the original input includes an abbreviation, do not "
                           "expand it into the full name."
        }
    }
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
        response.raise_for_status()     # 检查请求是否成功
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
            + content[-max_length // 2:]
        )

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: "
                "for instance run `pip install markdownify requests`."
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
        user_agent (str): 用于标识项目的自定义用户智能体字符串。根据维基百科API政策，这是必需的，
                详情请参阅：http://github.com/martin-majlis/Wikipedia-API/blob/master/README.rst
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
            "description": "The topic to search on Wikipedia. If the original input includes an abbreviation, "
                           "do not expand it into the full name.",
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
    description = "Performs a google web search for your query then returns a string of the top search results."
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The search query to perform. "
                "If the original input includes an abbreviation, do not expand it into the full name."
            )
        },
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi", cache_ttl: int = 86400, cache_dir: str = "./search_cache"):
        """
        初始化搜索工具，缓存功能默认开启
        :param provider: API提供商（serpapi/serper）
        :param cache_ttl: 缓存过期时间（秒），默认1天（86400秒）
        :param cache_dir: 缓存文件存储目录
        """
        super().__init__()
        
        # 缓存相关初始化（默认开启）
        self.cache_ttl = cache_ttl
        self.cache_dir = self._get_path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 原有初始化逻辑
        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = self._get_env(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def _get_path(self, path_str):
        """内部方法：获取Path对象（避免头部import）"""
        from pathlib import Path
        return Path(path_str)

    def _get_env(self, env_name):
        """内部方法：获取环境变量（避免头部import）"""
        import os
        return os.getenv(env_name)

    def _generate_cache_key(self, query: str, filter_year: int | None) -> str:
        """生成唯一的缓存键"""
        key_parts = [
            self.provider,
            query.replace(" ", "_").replace("/", "_").replace("?", "_"),
            str(filter_year) if filter_year else "no_year"
        ]
        return "_".join(key_parts) + ".json"

    def _get_cached_data(self, cache_key: str) -> dict | None:
        """读取缓存数据，检查是否过期"""
        import json
        import time
        
        cache_file = self.cache_dir / cache_key
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # 检查缓存是否过期
            current_time = time.time()
            if current_time - cache_data["timestamp"] > self.cache_ttl:
                if os.path.exists(cache_file):
                    try:
                        # 检查是否是文件（避免误删目录）
                        if os.path.isfile(cache_file):
                            os.remove(cache_file)
                            print(f"成功删除过期缓存文件: {cache_file}")
                        else:
                            # 如果是目录，给出提示并跳过（或根据需求删除目录）
                            print(f"警告: {cache_file} 是目录，而非文件，跳过删除")
                    except PermissionError:
                        # 权限不足异常处理
                        print(f"错误: 没有权限删除文件 {cache_file}，请检查文件权限或是否被占用")
                    except Exception as e:
                        # 捕获其他所有异常，避免程序崩溃
                        print(f"删除缓存文件 {cache_file} 失败: {str(e)}")
                else:
                    # 文件不存在时的友好提示（非错误）
                    print(f"缓存文件 {cache_file} 不存在，无需删除")
                return None
            
            return cache_data["data"]
        except (json.JSONDecodeError, KeyError, OSError):
            if cache_file.exists():
                import os
                os.remove(cache_file)
            return None

    def _save_to_cache(self, cache_key: str, data: dict):
        """保存数据到缓存"""
        import json
        import time
        
        cache_file = self.cache_dir / cache_key
        cache_data = {
            "timestamp": time.time(),
            "data": data
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def forward(self, query: str, filter_year: int | None = None) -> str:
        # 1. 生成缓存键（默认开启缓存）
        cache_key = self._generate_cache_key(query, filter_year)
        
        # 2. 尝试读取缓存
        cached_data = self._get_cached_data(cache_key)
        
        # 3. 缓存命中则直接使用
        if cached_data is not None:
            results = cached_data
        else:
            # 4. 缓存未命中，调用API
            import requests
            
            if self.provider == "serpapi":
                params = {
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                    "google_domain": "google.com",
                    "no_cache": False  # 启用SerpAPI服务器缓存
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
                self._save_to_cache(cache_key, results)  # 保存到本地缓存
            else:
                raise ValueError(f"API请求失败: {response.status_code} - {response.text}")

        # 原有结果处理逻辑
        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. "
                    f"Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return (
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, "
                f"or remove the year filter."
            )

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


class ArXivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "Primary arXiv paper search tool. "
        "ALWAYS use this tool to find relevant arXiv academic papers by keywords. "
        "Searches title+abstract+full text (same as official website). "
        "Supports precise filtering by submission year + month. "
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Core search keywords (e.g., AI regulation, machine learning). "
                           "Can be empty (auto-filled with category-related words).",
            "nullable": True
        },
        "categories": {
            "type": "string",
            "description": "Comma-separated arXiv subject categories (e.g., cs.CY, physics.soc-ph).",
            "nullable": True
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of papers to return (default: 10, MUST greater than 1).",
            "nullable": True
        },
        "submission_year": {
            "type": "integer",
            "description": "Filter papers by submission year (e.g., 2022, None if not given).",
            "nullable": True
        },
        "submission_month": {
            "type": "integer",
            "description": "Filter papers by submission month (1-12, None if not given).",
            "nullable": True
        },
        "submission_day": {
            "type": "integer",
            "description": "Filter papers by submission day (1-31, None if not given).",
            "nullable": True
        },
    }
    output_type = "string"

    def __init__(
            self,
            max_results: int = 10,
            sort_by: str = "submittedDate",
            sort_order: str = "descending",
            rate_limit: float = 0.33,
            retry_times: int = 2
    ):
        super().__init__()
        if max_results < 1 or max_results > 100:
            raise ValueError("max_results must be between 1 and 100 (arXiv API limit)")
        valid_sort_by = ["relevance", "lastUpdatedDate", "submittedDate"]
        if sort_by not in valid_sort_by:
            raise ValueError(f"sort_by must be one of: {valid_sort_by}")
        valid_sort_order = ["ascending", "descending"]
        if sort_order not in valid_sort_order:
            raise ValueError(f"sort_order must be one of: {valid_sort_order}")

        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.rate_limit = rate_limit
        self.retry_times = retry_times
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        self.base_url = "https://export.arxiv.org/api/query"
        self.category_default_keywords = {
            # 交叉学科/社会相关（仅保留核心通用词）
            "physics.soc-ph": "social physics complex systems",
            "cs.CY": "computers and society social computing",
            # AI/ML 核心分类（仅保留领域基础词）
            "cs.AI": "artificial intelligence",
            "cs.LG": "machine learning",
            "cs.CV": "computer vision",
            "stat.ML": "statistical machine learning",
            # 基础学科（仅保留学科核心名称）
            "math": "mathematics",
            "physics": "physics",
            "chemistry": "chemistry",
            "biology": "biology",
            "engineering": "engineering"
        }

    def _enforce_rate_limit(self) -> None:
        if not self.rate_limit:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _parse_arxiv_response(self, xml_response: str) -> List[dict]:
        """解析XML响应，提取核心字段（含精确年月）"""
        try:
            root = ET.fromstring(xml_response)
        except Exception as e:
            print(f"[ArXivSearchTool] XML parse error: {e}")
            return []

        papers = []
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}

        try:
            entries = root.findall('atom:entry', namespace)
        except SyntaxError:
            entries = root.findall('entry')

        print("\n[ArXivSearchTool] ===== API Response =====")
        print(f"[ArXivSearchTool] Total entries found: {len(entries)}")

        if not entries:
            print("[ArXivSearchTool] No entries found")
            return []

        for i, entry in enumerate(entries[:3], 1):
            try:
                title_elem = entry.find('atom:title', namespace)
                if title_elem is None:
                    title_elem = entry.find('title')
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"

                published_elem = entry.find('atom:published', namespace)
                if published_elem is None:
                    published_elem = entry.find('published')
                published = published_elem.text if published_elem is not None else "No date"

                print(f"[ArXivSearchTool] Raw paper {i}:")
                print(f"  Title: {title[:100]}..." if len(title) > 100 else f"  Title: {title}")
                print(f"  Published: {published}")
            except Exception as e:
                print(f"[ArXivSearchTool] Raw paper {i}: Failed to parse - {e}")

        for entry in entries:
            try:
                published_elem = entry.find('atom:published', namespace)
                if published_elem is None:
                    published_elem = entry.find('published')
                if published_elem is None or not published_elem.text:
                    continue

                publish_time = published_elem.text
                publish_parts = publish_time.split("T")[0].split("-")
                publish_year = int(publish_parts[0]) if len(publish_parts) >= 1 else 0
                publish_month = int(publish_parts[1]) if len(publish_parts) >= 2 else 0

                title_elem = entry.find('atom:title', namespace)
                if title_elem is None:
                    title_elem = entry.find('title')
                title = (
                    title_elem.text.strip().replace("\n", " ")
                    if title_elem is not None and title_elem.text
                    else "No title"
                )

                summary_elem = entry.find('atom:summary', namespace)
                if summary_elem is None:
                    summary_elem = entry.find('summary')
                abstract = (
                    summary_elem.text.strip().replace("\n", " ")
                    if summary_elem is not None and summary_elem.text
                    else "No abstract"
                )

                id_elem = entry.find('atom:id', namespace)
                if id_elem is None:
                    id_elem = entry.find('id')
                if id_elem is not None and id_elem.text:
                    arxiv_id = id_elem.text.split("/")[-1]
                    link = f"https://arxiv.org/pdf/{arxiv_id}"
                else:
                    arxiv_id = "Unknown"
                    link = "https://arxiv.org/pdf/"

                ''' 注释掉以避免覆盖上面的 pdf 地址
                link_elem = entry.find('atom:link[@rel="alternate"]', namespace)
                if link_elem is None:
                    for elem in entry.findall('link'):
                        if elem.get('rel') == 'alternate':
                            link_elem = elem
                            break
                if link_elem is not None:
                    link = link_elem.attrib.get("href", link)
                '''

                authors = []
                author_elems = entry.findall('atom:author', namespace)
                if not author_elems:
                    author_elems = entry.findall('author')
                for author in author_elems:
                    name_elem = author.find('atom:name', namespace)
                    if name_elem is None:
                        name_elem = author.find('name')
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                categories = []
                cat_elems = entry.findall('atom:category', namespace)
                if not cat_elems:
                    cat_elems = entry.findall('category')
                for cat in cat_elems:
                    if "term" in cat.attrib:
                        categories.append(cat.attrib["term"])

                paper = {
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "link": link,
                    "published": publish_time,
                    "publish_year": publish_year,
                    "publish_month": publish_month,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": categories
                }
                papers.append(paper)
            except Exception as e:
                print(f"[ArXivSearchTool] Failed to parse one paper: {e}")
                continue

        print(f"[ArXivSearchTool] Successfully parsed {len(papers)} papers")

        if papers:
            print("\n[ArXivSearchTool] ----- Parsed papers (first 3) -----")
            for i, paper in enumerate(papers[:3], 1):
                print(f"Paper {i}:")
                print(
                    f"  Title: {paper['title'][:100]}..."
                    if len(paper['title']) > 100
                    else f"  Title: {paper['title']}"
                )
                print(f"  Year: {paper['publish_year']}, Month: {paper['publish_month']:02d}")
                print(f"  Categories: {', '.join(paper['categories'][:3])}")
                print(f"  ID: {paper['arxiv_id']}")

        return papers

    def _get_default_keyword_for_category(self, categories: List[str]) -> str:
        if not categories:
            return "research"
        for cat in categories:
            cat_lower = cat.strip().lower()
            if cat_lower in self.category_default_keywords:
                return self.category_default_keywords[cat_lower]
            cat_prefix = cat_lower.split(".")[0]
            if cat_prefix in self.category_default_keywords:
                return self.category_default_keywords[cat_prefix]
        return "research"

    def _build_search_query(self, query, categories,
                            submission_year, submission_month, submission_day,
                            with_figures=False):
        search_parts = []

        # 日期
        if submission_year is not None:
            y = submission_year
            m = submission_month
            d = submission_day

            if m is not None and 1 <= m <= 12:
                if d is not None and 1 <= d <= 31:
                    start = f"{y:04d}{m:02d}{d:02d}"
                    end = start
                else:
                    if m == 2:
                        if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
                            last = 29
                        else:
                            last = 28
                    elif m in [4, 6, 9, 11]:
                        last = 30
                    else:
                        last = 31
                    start = f"{y:04d}{m:02d}01"
                    end = f"{y:04d}{m:02d}{last}"
            else:
                start = f"{y:04d}0101"
                end = f"{y:04d}1231"

            start_date = f"{start}0000"
            end_date = f"{end}2359"
            search_parts.append(f"submittedDate:[{start_date} TO {end_date}]")

        # 关键词
        base_query = (query or "").strip()
        if base_query:
            base_query = " ".join(base_query.split())

            if " " in base_query:
                # 短语匹配
                search_parts.append(f'all:"{base_query}"')
            else:
                search_parts.append(f"all:{base_query}")

        # 分类
        if categories:
            cat_parts = [f"cat:{c.strip()}" for c in categories if c.strip()]
            if cat_parts:
                search_parts.append("(" + " OR ".join(cat_parts) + ")")

        # 图表
        if with_figures:
            figure_query = '(all:"figure" OR all:"chart" OR all:"diagram" OR all:"visualization")'
            search_parts.append(figure_query)

        if search_parts:
            # 关键：用 AND 连接
            final_query = " AND ".join(search_parts)
        else:
            final_query = "all:*"

        print("[ArXivSearchTool] Final query:", final_query)
        return final_query

    def _filter_papers_by_year_month(
        self,
        papers: List[dict],
        target_year: Optional[int],
        target_month: Optional[int]
    ) -> List[dict]:
        """后过滤：按年份+月份筛选（作为API过滤的补充）"""
        if not papers:
            return papers

        if target_year is None:
            return papers

        filtered_by_year = [p for p in papers if p.get("publish_year") == target_year]

        if target_month is not None and 1 <= target_month <= 12:
            return [p for p in filtered_by_year if p.get("publish_month") == target_month]

        return filtered_by_year

    def _filter_papers_by_categories(self, papers: List[dict], target_cats: List[str]) -> List[dict]:
        """后过滤：按分类筛选（作为API过滤的补充）"""
        if not target_cats or not papers:
            return papers

        target_cats_lower = [c.strip().lower() for c in target_cats]
        filtered = []

        for paper in papers:
            paper_cats_lower = [c.lower() for c in paper.get("categories", [])]
            if any(tc in pc for tc in target_cats_lower for pc in paper_cats_lower):
                filtered.append(paper)

        return filtered

    def _core_search(self, query: str, categories: List[str], max_results: int,
                     submission_year: Optional[int], submission_month: Optional[int],
                     submission_day: Optional[int], with_figures: bool) -> str:
        print("\n[ArXivSearchTool] ===== Starting search =====")
        print(f"[ArXivSearchTool] Query: '{query}'")
        print(f"[ArXivSearchTool] Categories: {categories}")
        print(f"[ArXivSearchTool] Year: {submission_year}, Month: {submission_month}, "
              f"Day: {submission_day}")
        print(f"[ArXivSearchTool] Max results: {max_results}")

        # 构建查询
        search_query = self._build_search_query(
            query=query,
            categories=categories,
            submission_year=submission_year,
            submission_month=submission_month,
            submission_day=submission_day,
            with_figures=with_figures
        )

        api_max_results = min(max_results * 3, 50)
        print(f"[ArXivSearchTool] API max_results: {api_max_results}")

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": api_max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order
        }

        for attempt in range(self.retry_times + 1):
            try:
                self._enforce_rate_limit()
                print(f"\n[ArXivSearchTool] Attempt {attempt + 1}/{self.retry_times + 1}")
                print(f"[ArXivSearchTool] Request URL: {self.base_url}")
                print(f"[ArXivSearchTool] Request params: {params}")

                response = requests.get(self.base_url, params=params, timeout=30)
                print(f"[ArXivSearchTool] Response status: {response.status_code}")

                if response.status_code == 400 or response.status_code == 500:
                    print("[ArXivSearchTool] Bad request - trying without date filter")
                    # 移除日期过滤重试
                    fallback_parts = []

                    # 关键词
                    base_query = query.strip() or "research"
                    if len(base_query.split()) == 1:
                        fallback_parts.append(f"all:{base_query}")
                    else:
                        fallback_parts.append(f'all:"{base_query}"')

                    # 分类
                    if categories:
                        cat_parts = [f"cat:{cat}" for cat in categories if cat.strip()]
                        if cat_parts:
                            cat_query = "(" + "+OR+".join(cat_parts) + ")"
                            fallback_parts.append(cat_query)

                    params_without_date = {
                        "search_query": "+AND+".join(fallback_parts),
                        "start": 0,
                        "max_results": api_max_results,
                        "sortBy": self.sort_by,
                        "sortOrder": self.sort_order
                    }
                    response = requests.get(self.base_url, params=params_without_date, timeout=30)
                    print(f"[ArXivSearchTool] Retry without date - Response status: {response.status_code}")

                response.raise_for_status()

                raw_papers = self._parse_arxiv_response(response.text)

                if not raw_papers:
                    print("[ArXivSearchTool] No papers found from API")
                    if attempt == 0:
                        print("[ArXivSearchTool] Trying fallback query without quotes...")
                        fallback_query = query.strip() or "research"
                        params["search_query"] = f"all:{fallback_query}"
                        if categories:
                            cat_query = "+OR+".join([f"cat:{cat}" for cat in categories if cat])
                            params["search_query"] += f"+AND+({cat_query})"
                        continue
                    else:
                        return f"No arXiv papers found for query: '{search_query}'"

                # 后过滤
                filtered_papers = self._filter_papers_by_year_month(raw_papers, submission_year, submission_month)
                filtered_papers = self._filter_papers_by_categories(filtered_papers, categories)
                filtered_papers = filtered_papers[:max_results]

                print("\n[ArXivSearchTool] ===== Final Results =====")
                print(f"[ArXivSearchTool] Papers after all filters: {len(filtered_papers)}")

                if not filtered_papers:
                    filter_desc = []
                    if submission_year:
                        filter_desc.append(f"year={submission_year}")
                    if submission_month is not None and submission_year is not None:
                        filter_desc.append(f"month={submission_month}")

                    year_str = ", ".join(filter_desc) if filter_desc else "no year filter"
                    cat_str = ", ".join(categories) if categories else "no category filter"

                    return (f"No arXiv papers found after filtering.\n"
                            f"Query: '{search_query}'\n"
                            f"Filters: {year_str}, categories: {cat_str}\n"
                            f"Total papers retrieved from API: {len(raw_papers)}")

                formatted_results = ["## arXiv Search Results\n"]
                for idx, paper in enumerate(filtered_papers, 1):
                    authors_str = ", ".join(paper["authors"][:3])
                    if len(paper["authors"]) > 3:
                        authors_str += f" et al. ({len(paper['authors'])} authors total)"

                    categories_str = ", ".join(paper["categories"][:3])
                    if len(paper["categories"]) > 3:
                        categories_str += f" and {len(paper['categories'])-3} more"

                    publish_date = f"{paper['publish_year']}-{paper['publish_month']:02d}"

                    abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']

                    paper_entry = (
                        f"{idx}. **{paper['title']}**\n"
                        f"   - Authors: {authors_str}\n"
                        f"   - Categories: {categories_str}\n"
                        f"   - Submitted: {publish_date}\n"
                        f"   - Abstract: {abstract}\n"
                        f"   - Link: [{paper['arxiv_id']}]({paper['link']})\n"
                    )
                    formatted_results.append(paper_entry)

                print("[ArXivSearchTool] ===== Search completed =====\n")
                return "\n".join(formatted_results)

            except requests.exceptions.HTTPError as e:
                print(f"[ArXivSearchTool] HTTP Error: {e}")
                if attempt < self.retry_times:
                    wait_time = 2 * (attempt + 1)
                    print(f"[ArXivSearchTool] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return f"ArXiv API Error: {str(e)}"
            except requests.exceptions.Timeout:
                print("[ArXivSearchTool] Request timeout")
                return "ArXiv API Request Timed Out. Please try again later."
            except Exception as e:
                print(f"[ArXivSearchTool] Unexpected error: {e}")
                return f"Unexpected Error: {str(e)}"

    def forward(
            self,
            query: str = "",
            categories: Optional[str] = None,
            max_results: Optional[int] = None,
            submission_year: Optional[int] = None,
            submission_month: Optional[int] = None,
            submission_day: Optional[int] = None
    ) -> str:
        print("\n[ArXivSearchTool] ===== Tool called =====")
        print(f"[ArXivSearchTool] Input query: '{query}'")
        print(f"[ArXivSearchTool] Input categories: {categories}")
        print(f"[ArXivSearchTool] Input year: {submission_year}, month: {submission_month}, "
              f"day: {submission_day}")
        print(f"[ArXivSearchTool] Input max_results: {max_results}")

        results_count = max_results if max_results and 1 <= max_results <= 100 else self.max_results
        target_cats = categories.split(",") if categories else []

        print(f"[ArXivSearchTool] Using max_results: {results_count}")
        print(f"[ArXivSearchTool] Using categories: {target_cats}")

        return self._core_search(
            query=query,
            categories=target_cats,
            max_results=results_count,
            submission_year=submission_year,
            submission_month=submission_month,
            submission_day=submission_day,
            with_figures=False
        )


class USGSNASSpeciesSearchTool(Tool):

    name = "usgs_nas_species_search"

    description = (
        "Search the USGS Nonindigenous Aquatic Species (NAS) database by "
        "common name and return its content as a markdown string. "
        "Use this to get detailed species information from USGS NAS. "
        "If you want to get knowledges from USGS NAS database, call this tool with "
        "corresponding scientific name. "
        "Inputs ONLY use common name, DO NOT use scientific name. "
        "If no results are returned, check whether the common name is incorrect. "
        "If no results are returned, retry at least 3 times with different names. "
        "DO NOT run USGSNASSpeciesSearchToolin parallel with other search tools."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": "Common name of the species.",
        }
    }

    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length
        self.base_url_id = (
            "https://nas.er.usgs.gov/queries/SpeciesList.aspx?group=&genus=&species=&comname={}&Sortby=1"
        )
        self.base_url_fact = "https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID={}"

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2:]
        )

    def forward(self, query: str) -> str:
        try:
            import requests
            import re
            from urllib.parse import quote_plus

            encoded_query = quote_plus(query)
            url = self.base_url_id.format(encoded_query)

            response = requests.get(url, timeout=20)
            response.raise_for_status()

            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding

            html = response.text

            # 提取 speciesID + 物种名
            matches = re.findall(
                r"FactSheet\.aspx\?speciesID=(\d+)[^>]*>(.*?)</a>",
                html,
                re.IGNORECASE
            )

            if not matches:
                return f"No species found in USGS NAS database for query: {query}"

            markdown_content = ""
            for species_id, name in matches:
                # 拼接泛化URL（无任何特定物种硬编码）
                url = self.base_url_fact.format(species_id)

                # 复用现有requests逻辑，不新增import
                response = requests.get(url, timeout=20)
                response.raise_for_status()

                # 编码处理（复用现有逻辑）
                if response.encoding == 'ISO-8859-1':
                    response.encoding = response.apparent_encoding

                # HTML转Markdown（复用现有依赖，不新增import）
                from markdownify import markdownify
                import re
                markdown_content = markdown_content + markdownify(response.text).strip() + "\n\n"

            # 清理多余换行符
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            # 截断并返回纯网页内容（无结构化解析、无特定物种处理）
            return self._truncate_content(markdown_content, self.max_output_length)

        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: "
                "for instance run `pip install markdownify requests`."
            ) from e

        except requests.exceptions.Timeout:
            return f"Request to USGS NAS Species Search timed out after 20 seconds."

        except requests.exceptions.RequestException as e:
            return f"Error searching USGS NAS database: {str(e)}"

        except Exception as e:
            return f"Unexpected error during USGS NAS species search: {str(e)}"


class OSMNominatimGeocodeTool(Tool):
 
    name = "osm_nominatim_geocode"
 
    description = (
        "TRIGGER CONDITIONS (MUST CALL WHEN):\n"
        "1. User's question contains any of these keywords: ZIP code, postcode, postal code, latitude, longitude, 邮编\n"
        "2. User asks for geographic info of a place (city/state/country + zip/coordinates)\n"
        "\n"
        "STRICT RULES:\n"
        "- NEVER guess, estimate, or derive postcode/ZIP code/latitude/longitude from memory\n"
        "- NEVER answer geographic info without calling this tool first\n"
        "- MUST call this tool even if the address is incomplete (e.g., 'Manhattan' instead of 'Manhattan, NY')\n"
        "- If tool returns no results, use web search tool (DO NOT fallback to guessing)\n"
        "\n"
        "TOOL PURPOSE:\n"
        "Geocode an address/place name using OpenStreetMap Nominatim to get ACCURATE postcode (ZIP), "
        "latitude, longitude, city, state, country.\n"
        "\n"
        "INPUT RULE:\n"
        "query: Accept ANY place name/address (partial or full) related to the user's question "
        "(e.g., 'Manhattan NY', '北京市朝阳区')."
    )
    
    inputs = {
        "query": {
            "type": "string",
            "description": "Place name/address (partial or full) from user's question "
                           + "(no need to be 'full' - tool handles incomplete inputs)."
        }
    }
 
    output_type = "string"
 
    def __init__(self, max_output_length: int = 10000):
        super().__init__()
        self.max_output_length = max_output_length
        self.base_url = (
            "https://nominatim.openstreetmap.org/search"
            "?format=json&addressdetails=1&limit=1&q={}"
        )
 
    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2:]
        )
 
    def forward(self, query: str) -> str:
        try:
            import requests
            from urllib.parse import quote_plus
            import json
 
            encoded_query = quote_plus(query)
            url = self.base_url.format(encoded_query)
 
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              + "AppleWebKit/537.36 (KHTML, like Gecko) "
                              + "Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            }
 
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
 
            data = response.json()
 
            if not data:
                return f"No location found for query: {query}"
 
            result = data[0]
            address = result.get("address", {})
 
            postcode = address.get("postcode", "N/A")
            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or "N/A"
            )
            state = address.get("state", "N/A")
            country = address.get("country", "N/A")
 
            markdown_content = f"""
## Geocode Result
 
**Query:** {query}
 
**Display Name:** {result.get("display_name", "N/A")}
 
**Latitude:** {result.get("lat", "N/A")}
**Longitude:** {result.get("lon", "N/A")}
 
**Postcode (ZIP):** {postcode}
 
**City:** {city}
**State:** {state}
**Country:** {country}
"""
 
            return self._truncate_content(markdown_content.strip(), self.max_output_length)
 
        except ImportError as e:
            raise ImportError(
                "You must install package `requests` to run this tool: "
                "run `pip install requests`."
            ) from e
 
        except requests.exceptions.Timeout:
            return "Request to OpenStreetMap Nominatim timed out after 20 seconds."
 
        except requests.exceptions.RequestException as e:
            return f"Error querying OpenStreetMap Nominatim: {str(e)}"
 
        except Exception as e:
            return f"Unexpected error during geocoding: {str(e)}"
 

TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        WebSearchTool,
        DuckDuckGoSearchTool,
        GoogleSearchTool,
        ArXivSearchTool,
        USGSNASSpeciesSearchTool,
        OSMNominatimGeocodeTool,
    ]
}

__all__ = [
    "FinalAnswerTool",
    "UserInputTool",
    "WebSearchTool",    # 调用博査BochaAI接口
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "PythonInterpreterTool",
    "ArXivSearchTool",
    "USGSNASSpeciesSearchTool",
    "OSMNominatimGeocodeTool",
]
