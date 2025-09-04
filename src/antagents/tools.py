#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _convert_type_hints_to_json_schema,
    _get_json_schema_type,
    get_imports,
    get_json_schema,
)
from .agent_types import AgentAudio, AgentImage, handle_agent_input_types, handle_agent_output_types
from .tool_validation import MethodChecker, validate_tool_attributes
from .utils import (
    BASE_BUILTIN_MODULES,
    _is_package_available,
    get_source,
    instance_to_source,
    is_valid_name,
)


if TYPE_CHECKING:
    import mcp


logger = logging.getLogger(__name__)


def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class BaseTool(ABC):
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class Tool(BaseTool):
    """
    智能体使用函数的基础类。继承此类并实现 `forward` 方法以及以下类属性：

    - **description** (`str`) -- 对工具功能的简短描述，包括它期望的输入和将返回的输出。
      例如："这是一个从`url`下载文件的工具。它以`url`作为输入，并返回文件中包含的文本"。
    - **name** (`str`) -- 一个描述性名称，将在智能体提示中使用。例如 `"text-classifier"` 或 `"image_generator"`。
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]`) -- 输入期望的模态字典。
      包含一个`type`键和一个`description`键。
      这被`launch_gradio_demo`使用，或用于为您的工具创建漂亮的空间，也可以用于生成工具描述。
    - **output_type** (`type`) -- 工具输出的类型。这被`launch_gradio_demo`
      或用于为您的工具创建漂亮的空间使用，也可以用于生成工具描述。

    如果您的工具在使用前需要执行昂贵的操作（例如加载模型），您还可以重写方法 [`~Tool.setup`]。
    [`~Tool.setup`] 将在第一次使用工具时调用，但不会在实例化时调用。
    """

    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        # 验证类属性
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        # - 验证名称
        if not is_valid_name(self.name):
            raise Exception(
                f"Invalid tool name '{self.name}': must be a valid Python identifier and not a reserved keyword"
            )
        # 验证输入
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            # 将 input_types 作为列表获取，无论是字符串还是列表
            if isinstance(input_content["type"], str):
                input_types = [input_content["type"]]
            elif isinstance(input_content["type"], list):
                input_types = input_content["type"]
                # 检查所有元素是否为字符串
                if not all(isinstance(t, str) for t in input_types):
                    raise TypeError(
                        f"Input '{input_name}': when type is a list, all elements must be strings, got {input_content['type']}"
                    )
            else:
                raise TypeError(
                    f"Input '{input_name}': type must be a string or list of strings, got {type(input_content['type']).__name__}"
                )
            # 检查所有类型是否被授权
            invalid_types = [t for t in input_types if t not in AUTHORIZED_TYPES]
            if invalid_types:
                raise ValueError(f"Input '{input_name}': types {invalid_types} must be one of {AUTHORIZED_TYPES}")
        # 验证输出类型
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # 验证 forward 函数签名，除了使用"通用"签名的工具（SpaceToolWrapper, LangChainToolWrapper）
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)
            actual_keys = set(key for key in signature.parameters.keys() if key != "self")
            expected_keys = set(self.inputs.keys())
            if actual_keys != expected_keys:
                raise Exception(
                    f"In tool '{self.name}', 'forward' method parameters were {actual_keys}, but expected {expected_keys}. "
                    f"It should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # 此函数不会在缺少文档字符串时引发错误，与 get_json_schema 不同
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        # 处理可能作为单个字典传递的参数
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # 如果字典键匹配我们的输入参数，将其转换为 kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        """
        在此处重写此方法以执行任何昂贵的操作，这些操作需要在开始使用工具之前执行。例如加载大型模型。
        """
        self.is_initialized = True

    def to_code_prompt(self) -> str:
        args_signature = ", ".join(f"{arg_name}: {arg_schema['type']}" for arg_name, arg_schema in self.inputs.items())
        tool_signature = f"({args_signature}) -> {self.output_type}"
        tool_doc = self.description
        if self.inputs:
            args_descriptions = "\n".join(
                f"{arg_name}: {arg_schema['description']}" for arg_name, arg_schema in self.inputs.items()
            )
            args_doc = f"Args:\n{textwrap.indent(args_descriptions, '    ')}"
            tool_doc += f"\n\n{args_doc}"
        tool_doc = f'"""{tool_doc}\n"""'
        return f"def {self.name}{tool_signature}:\n{textwrap.indent(tool_doc, '    ')}"

    def to_tool_calling_prompt(self) -> str:
        return f"{self.name}: {self.description}\n    Takes inputs: {self.inputs}\n    Returns an output of type: {self.output_type}"

    def to_dict(self) -> dict:
        """返回表示工具的字典"""
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # 检查导入是否自包含
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # 如果工具是使用 '@tool' 装饰器创建的，它只有一个 forward 方法，因此只需获取其代码更简单
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (ValueError(f"SimpleTool validation failed for {self.name}:\n" + "\n".join(errors)))

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from antagents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()
            import re

            def add_self_argument(source_code: str) -> str:
                """如果不存在，将 'self' 添加为函数的第一个参数。"""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # 如果有其他参数
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # 如果工具不是通过 @tool 装饰器创建的，而是通过继承 Tool 类创建的
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "Cannot save objects created with from_space, from_langchain or from_gradio, as this would create errors."
                )

            validate_tool_attributes(self.__class__)

            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"antagents"}

        return {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

    @classmethod
    def from_dict(cls, tool_dict: dict[str, Any], **kwargs) -> "Tool":
        """
        从字典表示创建工具。

        参数:
            tool_dict (`dict[str, Any]`): 工具的字典表示。
            **kwargs: 传递给工具构造函数的额外关键字参数。

        返回:
            `Tool`: 工具对象。
        """
        if "code" not in tool_dict:
            raise ValueError("Tool dictionary must contain 'code' key with the tool source code")
        return cls.from_code(tool_dict["code"], **kwargs)

    def _get_gradio_app_code(self, tool_module_name: str = "tool") -> str:
        """获取 Gradio 应用代码。"""
        class_name = self.__class__.__name__
        return textwrap.dedent(
            f"""\
            from antagents import launch_gradio_demo
            from {tool_module_name} import {class_name}

            tool = {class_name}()
            launch_gradio_demo(tool)
            """
        )

    def _get_requirements(self) -> str:
        """获取 requirements。"""
        return "\n".join(self.to_dict()["requirements"])

    @staticmethod
    def from_gradio(gradio_tool):
        """
        从 gradio 工具创建 [`Tool`]。
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                self.inputs = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""} for key, value in func_args
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)


def launch_gradio_demo(tool: Tool):
    """
    为工具启动 gradio 演示。相应的工具类需要正确实现类属性 `inputs` 和 `output_type`。

    参数:
        tool (`Tool`): 要启动演示的工具。
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "boolean": gr.Checkbox,
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Textbox,
        "number": gr.Textbox,
    }

    def tool_forward(*args, **kwargs):
        return tool(*args, sanitize_inputs_outputs=True, **kwargs)

    tool_forward.__signature__ = inspect.signature(tool.forward)

    gradio_inputs = []
    for input_name, input_details in tool.inputs.items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[input_details["type"]]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[tool.output_type]
    gradio_output = output_gradio_component_class(label="Output")

    gr.Interface(
        fn=tool_forward,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool.name,
        description=tool.description,
        api_name=tool.name,
    ).launch()


def add_description(description):
    """
    为函数添加描述的装饰器。
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


class ToolCollection:
    """
    工具集合允许在智能体的工具箱中加载一组工具。

    集合可以从 Hub 的集合或 MCP 服务器加载，参见：
    - [`ToolCollection.from_mcp`]

    示例和用法请参见：[`ToolCollection.from_mcp`]
    """

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    @classmethod
    @contextmanager
    def from_mcp(
        cls, server_parameters: "mcp.StdioServerParameters" | dict, trust_remote_code: bool = False
    ) -> "ToolCollection":
        """自动从 MCP 服务器加载工具集合。

        此方法支持 Stdio、Streamable HTTP 和传统的 HTTP+SSE MCP 服务器。查看 `server_parameters`
        参数了解更多关于如何连接到每个 MCP 服务器的详细信息。

        注意：将生成一个单独的线程来运行处理 MCP 服务器的 asyncio 事件循环。

        参数:
            server_parameters (`mcp.StdioServerParameters` 或 `dict`):
                连接到 MCP 服务器的配置参数。可以是：

                - `mcp.StdioServerParameters` 的实例，用于通过子进程的标准输入/输出连接 Stdio MCP 服务器。

                - 一个 `dict`，至少包含：
                  - "url": 服务器的 URL。
                  - "transport": 要使用的传输协议，其中之一：
                    - "streamable-http": (推荐) Streamable HTTP 传输。
                    - "sse": 传统的 HTTP+SSE 传输（已弃用）。
                  如果省略 "transport"，则假定为传统的 "sse" 传输（将发出弃用警告）。

                <Deprecated version="1.17.0">
                HTTP+SSE 传输已弃用，未来行为将默认为 Streamable HTTP 传输。
                请显式传递 "transport" 键。
                </Deprecated>
            trust_remote_code (`bool`, *可选*, 默认为 `False`):
                是否信任从 MCP 服务器定义的代码执行。
                只有在您信任 MCP 服务器并了解在本地机器上运行远程代码的风险时，
                才应将此选项设置为 `True`。
                如果设置为 `False`，从 MCP 加载工具将失败。


        返回:
            ToolCollection: 工具集合实例。

        使用 Stdio MCP 服务器的示例:
        ```py
        >>> import os
        >>> from antagents import ToolCollection, ToolCallingAgent, OpenAIServerModel
        >>> from mcp import StdioServerParameters
        >>>
        >>> model = OpenAIServerModel(
        >>>     model_id=os.getenv("DEEPSEEK_MODEL_ID"),
        >>>     api_base=os.getenv("DEEPSEEK_URL"),
        >>>     api_key=os.getenv("DEEPSEEK_API_KEY")
        >>> )
        >>> server_parameters = StdioServerParameters(
        >>>     command="uvx",
        >>>     args=["--quiet", "pubmedmcp@0.1.3"],
        >>>     env={"UV_PYTHON": "3.12", **os.environ},
        >>> )

        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        >>>     agent = ToolCallingAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```

        使用 Streamable HTTP MCP 服务器的示例:
        ```py
        >>> with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True) as tool_collection:
        >>>     agent = ToolCallingAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```
        """
        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.antagents_adapter import antagentsAdapter
        except ImportError:
            raise ImportError(
                """Please install 'mcp' extra to use ToolCollection.from_mcp: `pip install "antagents[mcp]"`."""
            )
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                warnings.warn(
                    "Passing a dict as server_parameters without specifying the 'transport' key is deprecated. "
                    "For now, it defaults to the legacy 'sse' (HTTP+SSE) transport, but this default will change "
                    "to 'streamable-http' in version 1.20. Please add the 'transport' key explicitly. ",
                    FutureWarning,
                )
                transport = "sse"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        if not trust_remote_code:
            raise ValueError(
                "Loading tools from MCP requires you to acknowledge you trust the MCP server, "
                "as it will execute code on your local machine: pass `trust_remote_code=True`."
            )
        with MCPAdapt(server_parameters, antagentsAdapter()) as tools:
            yield cls(tools)


def tool(tool_function: Callable) -> Tool:
    """
    将函数转换为动态创建的 Tool 子类的实例。

    参数:
        tool_function (`Callable`): 要转换为 Tool 子类的函数。
            应为每个输入添加类型提示，并为输出添加类型提示。
            还应包含一个文档字符串，描述函数功能
            和一个 'Args:' 部分，其中描述每个参数。
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        if len(tool_json_schema["parameters"]["properties"]) == 0:
            tool_json_schema["return"] = {"type": "null"}
        else:
            raise TypeHintParsingException(
                "Tool return type not found: make sure your function has a return type hint!"
            )

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # 设置类属性
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]

    @wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        return tool_function(*args, **kwargs)

    # 将复制的函数绑定到 forward 方法
    SimpleTool.forward = staticmethod(wrapped_function)

    # 获取工具函数的签名参数
    sig = inspect.signature(tool_function)
    # - 添加 "self" 作为 tool_function 签名的第一个参数
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    # - 设置 forward 方法的签名
    SimpleTool.forward.__signature__ = new_sig

    # 创建并附加动态创建的工具类和 forward 方法的源代码
    # - 获取 tool_function 的源代码
    tool_source = inspect.getsource(tool_function)
    # - 移除工具装饰器和函数定义行
    tool_source_body = "\n".join(tool_source.split("\n")[2:])
    # - 取消缩进
    tool_source_body = textwrap.dedent(tool_source_body)
    # - 创建 forward 方法源代码，包括 def 行和缩进
    forward_method_source = f"def forward{str(new_sig)}:\n{textwrap.indent(tool_source_body, '    ')}"
    # - 创建类源代码
    class_source = (
        textwrap.dedent(f"""
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        """)
        + textwrap.indent(forward_method_source, "    ")  # 类方法的缩进
    )
    # - 在类和方法上存储源代码以供检查
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


def validate_tool_arguments(tool: Tool, arguments: Any) -> None:
    """根据工具的输入模式验证工具参数。

    检查所有提供的参数是否匹配工具的预期输入类型，并且
    所有必需的参数都存在。支持字典参数和单值参数
    用于具有一个输入参数的工具。

    参数:
        tool (`Tool`): 将使用其输入模式进行验证的工具。
        arguments (`Any`): 要验证的参数。可以是映射
            参数名称到值的字典，或单值用于具有一个输入的工具。


    抛出:
        ValueError: 如果参数不在工具的输入模式中，如果缺少必需的
            参数，或者参数值不匹配预期类型。
        TypeError: 如果参数具有无法转换的错误类型
            （例如，字符串而不是数字，不包括整数到数字的转换）。

    注意:
        - 支持从整数到数字的类型强制转换
        - 当在模式中明确标记时处理可空参数
        - 接受 "any" 类型作为匹配所有类型的通配符
    """
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if key not in tool.inputs:
                raise ValueError(f"Argument {key} is not in the tool's input schema")

            actual_type = _get_json_schema_type(type(value))["type"]
            expected_type = tool.inputs[key]["type"]
            expected_type_is_nullable = tool.inputs[key].get("nullable", False)

            # 如果类型匹配、是 "any" 或者是可空参数的 null，则类型有效
            if (
                (actual_type != expected_type if isinstance(expected_type, str) else actual_type not in expected_type)
                and expected_type != "any"
                and not (actual_type == "null" and expected_type_is_nullable)
            ):
                if actual_type == "integer" and expected_type == "number":
                    continue
                raise TypeError(f"Argument {key} has type '{actual_type}' but should be '{tool.inputs[key]['type']}'")

        for key, schema in tool.inputs.items():
            key_is_nullable = schema.get("nullable", False)
            if key not in arguments and not key_is_nullable:
                raise ValueError(f"Argument {key} is required")
        return None
    else:
        expected_type = list(tool.inputs.values())[0]["type"]
        if _get_json_schema_type(type(arguments))["type"] != expected_type and not expected_type == "any":
            raise TypeError(f"Argument has type '{type(arguments).__name__}' but should be '{expected_type}'")


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
    "launch_gradio_demo",
    "ToolCollection",
]