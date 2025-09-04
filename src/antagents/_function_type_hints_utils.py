#!/usr/bin/env python
# coding=utf-8

"""此模块包含专门从`transformers`仓库中提取的实用工具。

由于这些工具并非`transformers`特有，且`transformers`是一个重量级依赖项，
因此这些辅助函数被复制到了此处。

TODO: 将它们移动到`huggingface_hub`以避免代码重复。
"""

import inspect
import json
import re
import types
from collections.abc import Callable
from copy import copy
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


IMPORT_TO_PACKAGE_MAPPING = {
    "wikipediaapi": "wikipedia-api",
}


def get_package_name(import_name: str) -> str:
    """
    返回给定导入名称对应的包名称。

    参数:
        import_name (`str`): 要获取包名称的导入名称。

    返回:
        `str`: 给定导入名称对应的包名称。
    """
    return IMPORT_TO_PACKAGE_MAPPING.get(import_name, import_name)


def get_imports(code: str) -> list[str]:
    """
    提取代码中导入的所有库（不包括相对导入）。

    参数:
        code (`str`): 要检查的代码文本。

    返回:
        `list[str]`: 使用输入代码所需的所有包的列表。
    """
    # 过滤掉try/except块，以便在自定义代码中可以包含try/except导入
    code = re.sub(r"\s*try\s*:.*?except.*?:", "", code, flags=re.DOTALL)

    # 过滤掉is_flash_attn_2_available块下的导入，避免在仅CPU环境中出现导入问题
    code = re.sub(
        r"if is_flash_attn[a-zA-Z0-9_]+available\(\):\s*(from flash_attn\s*.*\s*)+",
        "",
        code,
        flags=re.MULTILINE,
    )

    # 形式为`import xxx`或`import xxx as yyy`的导入
    imports = re.findall(r"^\s*import\s+(\S+?)(?:\s+as\s+\S+)?\s*$", code, flags=re.MULTILINE)
    # 形式为`from xxx import yyy`的导入
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", code, flags=re.MULTILINE)
    # 仅保留顶级模块
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
    return [get_package_name(import_name) for import_name in set(imports)]


class TypeHintParsingException(Exception):
    """解析类型提示以生成JSON模式时抛出的异常"""


class DocstringParsingException(Exception):
    """解析文档字符串以生成JSON模式时抛出的异常"""


def get_json_schema(func: Callable) -> dict:
    """
    此函数根据给定函数的文档字符串和类型提示生成JSON模式。这主要用于将工具列表传递给聊天模板。
    JSON模式包含函数的名称和描述，以及其每个参数的名称、类型和描述。
    `get_json_schema()`要求函数有文档字符串，并且每个参数在文档字符串中有描述（使用标准的Google文档字符串格式）。
    它还要求所有函数参数都有有效的Python类型提示。

    虽然不是必需的，但也可以添加`Returns`块，它将被包含在模式中。这是可选的，因为大多数聊天模板会忽略函数的返回值。

    参数:
        func: 要生成JSON模式的函数。

    返回:
        包含函数JSON模式的字典。

    示例:
    ```python
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    '''
    >>>    return x * y
    >>>
    >>> print(get_json_schema(multiply))
    {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            },
            "required": ["x", "y"]
        }
    }
    ```

    这些模式的一般用途是用于生成支持工具描述的聊天模板的工具描述，例如：

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.utils import get_json_schema
    >>>
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    return x * y
    >>>    '''
    >>>
    >>> multiply_schema = get_json_schema(multiply)
    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    >>> messages = [{"role": "user", "content": "What is 179 x 4571?"}]
    >>> formatted_chat = tokenizer.apply_chat_template(
    >>>     messages,
    >>>     tools=[multiply_schema],
    >>>     chat_template="tool_use",
    >>>     return_dict=True,
    >>>     return_tensors="pt",
    >>>     add_generation_prompt=True
    >>> )
    >>> # 现在可以将格式化后的聊天传递给model.generate()
    ```

    每个参数描述还可以在末尾有一个可选的`(choices: ...)`块，例如`(choices: ["tea", "coffee"])`，
    这将被解析为模式中的`enum`字段。请注意，只有在行末时才能正确解析：

    ```python
    >>> def drink_beverage(beverage: str):
    >>>    '''
    >>>    A function that drinks a beverage
    >>>
    >>>    Args:
    >>>        beverage: The beverage to drink (choices: ["tea", "coffee"])
    >>>    '''
    >>>    pass
    >>>
    >>> print(get_json_schema(drink_beverage))
    ```
    {
        'name': 'drink_beverage',
        'description': 'A function that drinks a beverage',
        'parameters': {
            'type': 'object',
            'properties': {
                'beverage': {
                    'type': 'string',
                    'enum': ['tea', 'coffee'],
                    'description': 'The beverage to drink'
                    }
                },
            'required': ['beverage']
        }
    }
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"Cannot generate JSON schema for {func.__name__} because it has no docstring!"
        )
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = _parse_google_format_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:  # 我们允许缺少返回文档字符串，因为大多数模板会忽略它
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'"
            )
        desc = param_descriptions[arg]
        enum_choices = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema["enum"] = [c.strip() for c in json.loads(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema["description"] = desc

    output = {"name": func.__name__, "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


# 提取文档字符串的初始段，包含函数描述
description_re = re.compile(r"^(.*?)(?=\n\s*(Args:|Returns:|Raises:)|\Z)", re.DOTALL)
# 从文档字符串中提取Args:块
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# 将Args:块分割为单个参数
args_split_re = re.compile(
    r"(?:^|\n)"  # 匹配args块的开始或换行
    r"\s*(\w+)\s*(?:\([^)]*?\))?:\s*"  # 捕获参数名称（忽略类型）并去除空格
    r"(.*?)\s*"  # 捕获参数描述，可以跨多行，并去除尾部空格
    r"(?=\n\s*\w+\s*(?:\([^)]*?\))?:|\Z)",  # 当遇到下一个参数（带或不带类型）或块结束时停止
    re.DOTALL | re.VERBOSE,
)
# 从文档字符串中提取Returns:块（如果存在）。请注意，大多数聊天模板会忽略返回类型/文档！
returns_re = re.compile(
    r"\n\s*Returns:\n\s*"
    r"(?:[^)]*?:\s*)?"  # 如果存在返回类型则忽略
    r"(.*?)"  # 捕获返回描述
    r"[\n\s]*(Raises:|\Z)",
    re.DOTALL,
)


def _parse_google_format_docstring(
    docstring: str,
) -> tuple[str | None, dict | None, str | None]:
    """
    解析Google风格的文档字符串以提取函数描述、参数描述和返回描述。

    参数:
        docstring (str): 要解析的文档字符串。

    返回:
        函数描述、参数和返回描述。
    """

    # 提取各部分
    description_match = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)

    # 清理并存储各部分
    description = description_match.group(1).strip() if description_match else None
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    # 将参数解析为字典
    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])  # 移除空行
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0]: re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
    else:
        args_dict = {}

    return description, args_dict, returns


def _convert_type_hints_to_json_schema(func: Callable, error_on_missing_type_hints: bool = True) -> dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty and error_on_missing_type_hints:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param_name not in properties:
            properties[param_name] = {}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["nullable"] = True

    # 返回：多类型联合 -> 视为any
    if (
        "return" in properties
        and (return_type := properties["return"].get("type"))
        and not isinstance(return_type, str)
    ):
        properties["return"]["type"] = "any"

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def _parse_type_hint(hint: type) -> dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException(
                "Couldn't parse this type hint, likely due to a custom class or object: ",
                hint,
            )

    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        return _parse_union_type(args)

    elif origin is list:
        if not args:
            return {"type": "array"}
        else:
            # 列表只能有一个类型参数，因此递归解析它
            return {"type": "array", "items": _parse_type_hint(args[0])}

    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace('typing.', '')} is a Tuple with a single element, which "
                "we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain "
                "more than one element, we recommend "
                "using a List[] type instead, or if it really is a single element, remove the Tuple[] wrapper and just "
                "pass the element directly."
            )
        if ... in args:
            raise TypeHintParsingException(
                "Conversion of '...' is not supported in Tuple type hints. "
                "Use List[] types for variable-length"
                " inputs instead."
            )
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}

    elif origin is dict:
        # JSON中等效于dict的是'object'，它要求所有键都是字符串
        # 但是，我们可以使用"additionalProperties"指定字典值的类型
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out

    elif origin is Literal:
        literal_types = set(type(arg) for arg in args)
        final_type = _parse_union_type(literal_types)

        # None字面值由_parse_union_type设置的'nullable'字段表示
        final_type.update({"enum": [arg for arg in args if arg is not None]})
        return final_type

    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object: ", hint)


def _parse_union_type(args: tuple[Any, ...]) -> dict:
    subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
    if len(subtypes) == 1:
        # 单个非null类型可以直接表示
        return_dict = subtypes[0]
    elif all(isinstance(subtype["type"], str) for subtype in subtypes):
        # 基本类型的联合可以在模式中表示为列表
        return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
    else:
        # 更复杂类型的联合需要"anyOf"
        return_dict = {"anyOf": subtypes}
    if type(None) in args:
        return_dict["nullable"] = True
    return return_dict


_BASE_TYPE_MAPPING = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    Any: {"type": "any"},
    types.NoneType: {"type": "null"},
}


def _get_json_schema_type(param_type: type) -> dict[str, str]:
    if param_type in _BASE_TYPE_MAPPING:
        return copy(_BASE_TYPE_MAPPING[param_type])
    if str(param_type) == "Image":
        from PIL.Image import Image

        if param_type == Image:
            return {"type": "image"}
    if str(param_type) == "Tensor":
        try:
            from torch import Tensor

            if param_type == Tensor:
                return {"type": "audio"}
        except ModuleNotFoundError:
            pass
    return {"type": "object"}