#!/usr/bin/env python
# coding=utf-8

import ast
import base64
import importlib.metadata
import importlib.util
import inspect
import json
import keyword
import os
import re
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import jinja2


if TYPE_CHECKING:
    from antagents.memory import AgentLogger


__all__ = ["AgentError"]


@lru_cache
def _is_package_available(package_name: str) -> bool:
    """检查指定包是否可用"""
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


def escape_code_brackets(text: str) -> str:
    """转义代码段中的方括号，同时保留Rich样式标签"""
    def replace_bracketed_content(match):
        content = match.group(1)
        cleaned = re.sub(
            r"bold|red|green|blue|yellow|magenta|cyan|white|black|italic|dim|\s|#[0-9a-fA-F]{6}", "", content
        )
        return f"\\[{content}\\]" if cleaned.strip() else f"[{content}]"

    return re.sub(r"\[([^\]]*)\]", replace_bracketed_content, text)


def decode_unicode_escapes(text):
    def decode_string(s):
        # 安全解码Unicode转义，不影响已编码的中文字符
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except:
                return match.group(0)
        
        # 只处理真正的Unicode转义序列
        return re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode, s)
    
    # 首先尝试解析为JSON
    try:
        parsed = json.loads(text)
        
        # 递归处理JSON中的所有字符串
        def process_obj(obj):
            if isinstance(obj, str):
                return decode_string(obj)
            elif isinstance(obj, dict):
                return {k: process_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process_obj(x) for x in obj]
            else:
                return obj
        
        processed = process_obj(parsed)
        return json.dumps(processed, ensure_ascii=False)
    
    except json.JSONDecodeError:
        # 如果不是JSON，只处理字符串中的Unicode转义
        return decode_string(text)


class AgentError(Exception):
    """智能体相关异常的基类"""

    def __init__(self, message, logger: "AgentLogger"):
        super().__init__(message)
        self.message = message
        logger.log_error(message)

    def dict(self) -> dict[str, str]:
        return {"type": self.__class__.__name__, "message": str(self.message)}


class AgentParsingError(AgentError):
    """智能体解析错误时抛出的异常"""
    pass


class AgentExecutionError(AgentError):
    """智能体执行错误时抛出的异常"""
    pass


class AgentMaxStepsError(AgentError):
    """智能体执行步骤超过限制时抛出的异常"""
    pass


class AgentToolCallError(AgentExecutionError):
    """工具调用参数不正确时抛出的异常"""
    pass


class AgentToolExecutionError(AgentExecutionError):
    """工具执行错误时抛出的异常"""
    pass


class AgentGenerationError(AgentError):
    """智能体生成错误时抛出的异常"""
    pass


def make_json_serializable(obj: Any) -> Any:
    """递归函数，使对象可JSON序列化"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # 尝试将看起来像JSON对象/数组的字符串解析为JSON
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # 对于自定义对象，将其__dict__转换为可序列化格式
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # 对于其他类型，转换为字符串
        return str(obj)


def parse_json_blob(json_blob: str) -> tuple[dict[str, str], str]:
    """从输入中提取JSON blob并返回JSON数据和输入的其余部分"""
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_str = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_str, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    """截断内容，使其不超过最大长度"""
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


class ImportFinder(ast.NodeVisitor):
    """AST节点访问器，用于查找导入的包"""
    def __init__(self):
        self.packages = set()

    def visit_Import(self, node):
        for alias in node.names:
            # 获取基础包名（点号前的部分）
            base_package = alias.name.split(".")[0]
            self.packages.add(base_package)

    def visit_ImportFrom(self, node):
        if node.module:  # 处理"from x import y"语句
            # 获取基础包名（点号前的部分）
            base_package = node.module.split(".")[0]
            self.packages.add(base_package)


def instance_to_source(instance, base_cls=None):
    """将实例转换为其类的源代码表示"""
    cls = instance.__class__
    class_name = cls.__name__

    # 开始构建类代码行
    class_lines = []
    if base_cls:
        class_lines.append(f"class {class_name}({base_cls.__name__}):")
    else:
        class_lines.append(f"class {class_name}:")

    # 如果存在文档字符串且与基类不同，则添加
    if cls.__doc__ and (not base_cls or cls.__doc__ != base_cls.__doc__):
        class_lines.append(f'    """{cls.__doc__}"""')

    # 添加类级别属性
    class_attrs = {
        name: value
        for name, value in cls.__dict__.items()
        if not name.startswith("__")
        and not name == "_abc_impl"
        and not callable(value)
        and not (base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value)
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            # 多行值
            if "\n" in value:
                escaped_value = value.replace('"""', r"\"\"\"")  # 转义三引号
                class_lines.append(f'    {name} = """{escaped_value}"""')
            else:
                class_lines.append(f"    {name} = {json.dumps(value)}")
        else:
            class_lines.append(f"    {name} = {repr(value)}")

    if class_attrs:
        class_lines.append("")

    # 添加方法
    methods = {
        name: func.__wrapped__ if hasattr(func, "__wrapped__") else func
        for name, func in cls.__dict__.items()
        if callable(func)
        and (
            not base_cls
            or not hasattr(base_cls, name)
            or (
                isinstance(func, (staticmethod, classmethod))
                or (getattr(base_cls, name).__code__.co_code != func.__code__.co_code)
            )
        )
    }

    for name, method in methods.items():
        method_source = get_source(method)
        # 清理缩进
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(["    " + line if line.strip() else line for line in method_lines])
        class_lines.append(method_source)
        class_lines.append("")

    # 使用ImportFinder查找所需的导入
    import_finder = ImportFinder()
    import_finder.visit(ast.parse("\n".join(class_lines)))
    required_imports = import_finder.packages

    # 构建带有导入的最终代码
    final_lines = []

    # 如果需要，添加基类导入
    if base_cls:
        final_lines.append(f"from {base_cls.__module__} import {base_cls.__name__}")

    # 添加发现的导入
    for package in required_imports:
        final_lines.append(f"import {package}")

    if final_lines:  # 在导入后添加空行
        final_lines.append("")

    # 添加类代码
    final_lines.extend(class_lines)

    return "\n".join(final_lines)


def get_source(obj) -> str:
    """获取类或可调用对象（如函数、方法）的源代码
    首先尝试使用`inspect.getsource`获取源代码
    在动态环境（如Jupyter、IPython）中，如果失败，
    则回退到从当前交互式shell会话中检索源代码

    参数:
        obj: 类或可调用对象（如函数、方法）

    返回:
        str: 对象的源代码，去除缩进和前后空格

    抛出:
        TypeError: 如果对象不是类或可调用对象
        OSError: 如果无法从任何源检索源代码
        ValueError: 如果在IPython历史记录中找不到源代码

    注意:
        TODO: 处理Python标准REPL
    """
    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")

    inspect_error = None
    try:
        # 处理动态创建的类
        source = getattr(obj, "__source__", None) or inspect.getsource(obj)
        return dedent(source).strip()
    except OSError as e:
        # 保留异常，如果所有后续方法都失败则抛出
        inspect_error = e
    try:
        import IPython

        shell = IPython.get_ipython()
        if not shell:
            raise ImportError("No active IPython shell found")
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history")
    except ImportError:
        # IPython不可用，抛出原始inspect错误
        raise inspect_error
    except ValueError as e:
        # IPython可用但找不到源代码，抛出错误
        raise e from inspect_error


def encode_image_base64(image):
    """将图像编码为base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_image_url(base64_image):
    """从base64编码的图像创建URL"""
    return f"data:image/png;base64,{base64_image}"


def make_init_file(folder: str | Path):
    """创建__init__.py文件"""
    os.makedirs(folder, exist_ok=True)
    # 创建__init__
    with open(os.path.join(folder, "__init__.py"), "w"):
        pass


def is_valid_name(name: str) -> bool:
    """检查名称是否是有效的Python标识符且不是关键字"""
    return name.isidentifier() and not keyword.iskeyword(name) if isinstance(name, str) else False


AGENT_GRADIO_APP_TEMPLATE = """import yaml
import os
from antagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

# 获取当前目录路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

{% for tool in tools.values() -%}
from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
{% endfor %}
{% for managed_agent in managed_agents.values() -%}
from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
{% endfor %}

model = {{ agent_dict['model']['class'] }}(
{% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
    {{ key }}={{ agent_dict['model']['data'][key]|repr }},
{% endfor %})

{% for tool in tools.values() -%}
{{ tool.name }} = {{ tool.name | camelcase }}()
{% endfor %}

with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

{{ agent_name }} = {{ class_name }}(
    model=model,
    tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
    managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
    {% for attribute_name, value in agent_dict.items() if attribute_name not in ["class", "model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
    {{ attribute_name }}={{ value|repr }},
    {% endfor %}prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI({{ agent_name }}).launch()
""".strip()


def create_agent_gradio_app_template():
    """创建Gradio应用模板"""
    env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
    env.filters["repr"] = repr
    env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
    return env.from_string(AGENT_GRADIO_APP_TEMPLATE)


class RateLimiter:
    """简单的速率限制器，强制连续请求之间的最小延迟

    这个类对于限制操作速率（如API请求）很有用，
    通过确保基于期望的每分钟请求数，`throttle()`调用之间至少间隔给定时间

    如果没有指定速率（即`requests_per_minute`为None），则禁用速率限制，
    `throttle()`变为无操作

    参数:
        requests_per_minute (`float | None`): 每分钟允许的最大请求数
            使用`None`禁用速率限制
    """

    def __init__(self, requests_per_minute: float | None = None):
        self._enabled = requests_per_minute is not None
        self._interval = 60.0 / requests_per_minute if self._enabled else 0.0
        self._last_call = 0.0

    def throttle(self):
        """暂停执行以遵守速率限制（如果启用）"""
        if not self._enabled:
            return
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.time()