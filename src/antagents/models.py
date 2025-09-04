import json
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from .monitoring import TokenUsage
from .tools import Tool
from .utils import RateLimiter, _is_package_available, encode_image_base64, make_image_url, parse_json_blob


logger = logging.getLogger(__name__)


# 添加token估算器（简化版）
def estimate_tokens_from_messages(messages: list[dict]) -> int:
    """
    估算消息列表的大致token数量
    
    Args:
        messages: 消息列表，每个消息是包含role和content的字典
        
    Returns:
        int: 估算的token数量
    """
    total_tokens = 0
    for message in messages:
        # 计算消息内容的token数（近似：1个token ≈ 4个英文字符或2个中文字符）
        content = message.get('content', '')
        if isinstance(content, str):
            # 简单估算：英文字符数/4 + 中文字符数/2
            english_chars = len(re.findall(r'[a-zA-Z0-9\s\.,!?;:\'"\-]', content))
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            content_tokens = english_chars // 4 + chinese_chars // 2
        elif isinstance(content, list):
            # 多模态内容：估算所有文本部分
            content_tokens = 0
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text = item['text']
                    english_chars = len(re.findall(r'[a-zA-Z0-9\s\.,!?;:\'"\-]', text))
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
                    content_tokens += english_chars // 4 + chinese_chars // 2
        else:
            content_tokens = 100  # 默认值
            
        # 角色和格式开销（约10个token）
        total_tokens += content_tokens + 10
        
    return total_tokens


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallFunction:
    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallFunction
    id: str
    type: str

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


@dataclass
class ChatMessage:
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # 存储来自API的原始输出
    token_usage: TokenUsage | None = None

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        return get_dict_from_nested_dataclasses(self)

    def render_as_markdown(self) -> str:
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps({"tool": tool.function.name, "arguments": tool.function.arguments})
                    for tool in self.tool_calls
                ]
            )
        return rendered


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageToolCallStreamDelta:
    """表示生成过程中工具调用的流式增量。"""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ChatMessageToolCallFunction | None = None


@dataclass
class ChatMessageStreamDelta:
    content: str | None = None
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None
    token_usage: TokenUsage | None = None


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """
    将流式增量列表聚合成单个流式增量。
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:  # 通常一次应该只有一个调用
                # 如果需要，扩展accumulated_tool_calls列表以容纳新的工具调用
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        )
                    # 更新特定索引处的工具调用
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if tool_call_delta.function.name and len(tool_call_delta.function.name) > 0:
                            tool_call.function.name = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += tool_call_delta.function.arguments
                else:
                    raise ValueError(f"Tool call index is not provided in tool delta: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: list[ChatMessage | dict],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """
    创建用于输入给LLM的消息列表。这些消息是字典格式，并且与transformers LLM聊天模板兼容。
    相同角色的连续消息将被合并为单个消息。

    参数:
        message_list (`list[ChatMessage | dict]`): 聊天消息列表。允许混合类型。
        role_conversions (`dict[MessageRole, MessageRole]`, *可选*): 用于转换角色的映射。
        convert_images_to_image_urls (`bool`, 默认 `False`): 是否将图像转换为图像URL。
        flatten_messages_as_text (`bool`, 默认 `False`): 是否将消息扁平化为文本。
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # 避免修改原始列表
    for message in message_list:
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        # 如果需要，编码图像
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
            assert isinstance(message.content, list), "Error: wrong content:" + str(message.content)
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                for el in message.content:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # 合并连续的文本消息，而不是创建新的消息
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Key {tool_name_key=} not found in the generated tool call. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    检查模型是否支持`stop`参数。

    不支持推理模型openai/o3和openai/o4-mini（及其版本变体）。

    参数:
        model_id (`str`): 模型标识符（例如"openai/o3", "o4-mini-2025-04-16"）

    返回:
        bool: 如果模型支持stop参数则返回True，否则返回False
    """
    model_name = model_id.split("/")[-1]
    # o3和o4-mini（包括版本变体，如o3-2025-04-16）不支持stop参数
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*)$"
    return not re.match(pattern, model_name)


class Model:
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        max_token_limit: int = 100000,  # 添加最大token限制，默认100K
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self._last_input_token_count: int | None = None
        self._last_output_token_count: int | None = None
        self.model_id: str | None = model_id
        self.max_token_limit = max_token_limit  # 存储最大token限制

    @property
    def last_input_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_input_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.input_tokens instead.",
            FutureWarning,
        )
        return self._last_input_token_count

    @property
    def last_output_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_output_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.output_tokens instead.",
            FutureWarning,
        )
        return self._last_output_token_count

    def _check_token_limit(self, messages: list[dict], estimated_output_tokens: int = 0) -> None:
        """
        检查估算的token数量是否超过限制
        
        Args:
            messages: 清理后的消息列表
            estimated_output_tokens: 预估的输出token数量
            
        Raises:
            ValueError: 如果估算的token数量超过限制
        """
        # 估算输入token数量
        estimated_input_tokens = estimate_tokens_from_messages(messages)
        
        # 估算总token数量（输入+输出）
        total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
        
        # 检查是否超过限制
        if total_estimated_tokens > self.max_token_limit:
            raise ValueError(
                f"Estimated token count ({total_estimated_tokens}) exceeds maximum limit "
                f"of {self.max_token_limit}. Input tokens: {estimated_input_tokens}, "
                f"Estimated output tokens: {estimated_output_tokens}"
            )
        
        logger.info(f"Estimated tokens: input={estimated_input_tokens}, "
                   f"output={estimated_output_tokens}, total={total_estimated_tokens}")

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",  # 可配置的tool_choice参数
        estimated_output_tokens: int = 1000,  # 默认预估输出token数
        **kwargs,
    ) -> dict[str, Any]:
        """
        准备模型调用所需的参数，处理参数优先级。

        参数优先级从高到低：
        1. 显式传递的kwargs
        2. 特定参数（stop_sequences, response_format等）
        3. self.kwargs中的默认值
        """
        # 清理并标准化消息列表
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        
        # 检查token限制
        self._check_token_limit(messages_as_dicts, estimated_output_tokens)
        
        # 使用self.kwargs作为基础配置
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
        }

        # 处理特定参数
        if stop_sequences is not None:
            # 某些模型不支持stop参数
            if supports_stop_parameter(self.model_id or ""):
                completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # 处理tools参数
        if tools_to_call_from:
            tools_config = {
                "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        # 最后，使用传入的kwargs覆盖所有设置
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """处理输入消息并返回模型的响应。

        参数:
            messages (`list[dict[str, str | list[dict]] | list[ChatMessage]`):
                要处理的消息字典列表。每个字典应具有结构`{"role": "user/system", "content": "message content"}`。
            stop_sequences (`List[str]`, *可选*):
                字符串列表，如果在模型输出中遇到将停止生成。
            response_format (`dict[str, str]`, *可选*):
                用于模型响应的响应格式。
            tools_to_call_from (`List[Tool]`, *可选*):
                模型可用于生成响应的工具列表。
            **kwargs:
                传递给底层模型的额外关键字参数。

        返回:
            `ChatMessage`: 包含模型响应的聊天消息对象。
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """有时API不将工具调用作为特定对象返回，因此我们需要解析它。"""
        message.role = MessageRole.ASSISTANT  # 如果需要，覆盖角色
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        """
        将模型转换为JSON兼容的字典。
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
            "max_token_limit": self.max_token_limit,  # 包含token限制设置
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        return cls(**{k: v for k, v in model_dictionary.items()})


class ApiModel(Model):
    """
    基于API的语言模型的基类。

    该类作为实现与外部API交互的模型的基础。它处理管理模型ID、
    自定义角色映射和API客户端连接的通用功能。

    参数:
        model_id (`str`):
            用于API的模型标识符。
        custom_role_conversions (`dict[str, str`], **可选**):
            用于在内部角色名称和API特定角色名称之间转换的映射。默认为None。
        client (`Any`, **可选**):
            预配置的API客户端实例。如果未提供，将创建默认客户端。默认为None。
        requests_per_minute (`float`, **可选**):
            每分钟的请求速率限制。
        **kwargs: 传递给父类的额外关键字参数。
    """

    def __init__(
        self,
        model_id: str,
        custom_role_conversions: dict[str, str] | None = None,
        client: Any | None = None,
        requests_per_minute: float | None = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()
        self.rate_limiter = RateLimiter(requests_per_minute)

    def create_client(self):
        """为特定服务创建API客户端。"""
        raise NotImplementedError("Subclasses must implement this method to create a client")

    def _apply_rate_limit(self):
        """在发起API调用之前应用速率限制。"""
        self.rate_limiter.throttle()


class OpenAIServerModel(ApiModel):
    """此模型连接到OpenAI兼容的API服务器。

    参数:
        model_id (`str`):
            服务器上使用的模型标识符（例如"gpt-3.5-turbo"）。
        api_base (`str`, *可选*):
            OpenAI兼容API服务器的基础URL。
        api_key (`str`, *可选*):
            用于身份验证的API密钥。
        organization (`str`, *可选*):
            用于API请求的组织。
        project (`str`, *可选*):
            用于API请求的项目。
        client_kwargs (`dict[str, Any]`, *可选*):
            传递给OpenAI客户端的额外关键字参数（如organization, project, max_retries等）。
        custom_role_conversions (`dict[str, str]`, *可选*):
            用于转换消息角色的自定义映射。
            对于不支持特定消息角色（如"system"）的特定模型非常有用。
        flatten_messages_as_text (`bool`, 默认 `False`):
            是否将消息扁平化为文本。
        **kwargs:
            传递给OpenAI API的额外关键字参数。
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'antagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        # 估算输出token数（基于max_tokens参数）
        estimated_output_tokens = kwargs.get('max_tokens', 1000)
        
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            estimated_output_tokens=estimated_output_tokens,
            **kwargs,
        )
        # print(json.dumps(completion_kwargs, indent=2, ensure_ascii=False))
        for msg in completion_kwargs.get('messages', []):
            if isinstance(msg.get('content'), list) and len(msg['content']) > 0:
                first_content = msg['content'][0]
                if isinstance(first_content, dict) and 'text' in first_content:
                    msg['content'] = first_content['text']
        self._apply_rate_limit()
        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # 估算输出token数（基于max_tokens参数）
        estimated_output_tokens = kwargs.get('max_tokens', 1000)
        
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            estimated_output_tokens=estimated_output_tokens,
            **kwargs,
        )
        # print(json.dumps(completion_kwargs, indent=2, ensure_ascii=False))
        for msg in completion_kwargs.get('messages', []):
            if isinstance(msg.get('content'), list) and len(msg['content']) > 0:
                first_content = msg['content'][0]
                if isinstance(first_content, dict) and 'text' in first_content:
                    msg['content'] = first_content['text']
        self._apply_rate_limit()
        response = self.client.chat.completions.create(**completion_kwargs)

        # 据报道，使用OpenRouter时，`response.usage`在某些情况下可能为None：参见GH-1401
        self._last_input_token_count = getattr(response.usage, "prompt_tokens", 0)
        self._last_output_token_count = getattr(response.usage, "completion_tokens", 0)
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


class GeminiServerModel(ApiModel):
    """Gemini API服务模型封装类
    
    提供与Google Gemini API的交互能力，支持文本和多模态内容生成
    
    特性：
    - 支持流式和非流式生成
    - 自动处理角色转换（将标准ChatML角色转换为Gemini支持的角色）
    - 支持多模态输入（文本+图像）
    - 内置速率限制
    - 支持工具调用转换
    
    典型用法：
        >>> model = GeminiServerModel(
        ...     model_id="gemini-pro",
        ...     api_key="your_api_key"
        ... )
        >>> response = model.generate([{"role":"user","content":"Hello"}])
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        """初始化Gemini模型实例
        
        Args:
            model_id: 模型标识符（如"gemini-pro"）
            api_base: API基础URL（可选，默认为官方端点）
            api_key: Gemini API密钥
            project: Google Cloud项目ID（可选）
            client_kwargs: 传递给GenAI客户端的额外参数
            custom_role_conversions: 自定义角色转换映射
            flatten_messages_as_text: 是否将消息内容扁平化为纯文本
            **kwargs: 其他模型生成参数（如temperature等）
        """
        # 客户端配置（包含认证和连接参数）
        self.client_kwargs = {
            **(client_kwargs or {}),  # 用户自定义参数
            "api_key": api_key,  # API认证密钥
            "transport": "rest",  # 使用REST传输协议
            "client_options": {"api_endpoint": api_base} if api_base else None,  # 自定义API端点
        }
        
        # 设置Google Cloud项目（如果提供）
        if project:
            self.client_kwargs["project"] = project

        # 默认角色转换规则（将各种ChatML角色映射为Gemini支持的两种角色）
        default_role_conversions = {
            "system": "user",    # 系统提示 -> 用户角色
            "assistant": "model",  # 助手回复 -> 模型角色
            "tool": "user",      # 工具响应 -> 用户角色
            "tool-call": "model", # 工具调用 -> 模型角色
            "tool-response": "user",
        }
        
        # 合并用户自定义的角色转换规则
        if custom_role_conversions:
            default_role_conversions.update(custom_role_conversions)

        # 初始化父类（ApiModel）
        super().__init__(
            model_id=model_id,
            custom_role_conversions=default_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self) -> Any:
        """创建并配置Gemini客户端
        
        Returns:
            google.generativeai模块实例
            
        Raises:
            ModuleNotFoundError: 未安装google-generativeai包时抛出
        """
        try:
            import google.generativeai as genai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "请先安装google-generativeai包: pip install google-generativeai"
            ) from e

        # 使用配置参数初始化客户端
        genai.configure(**self.client_kwargs)
        return genai

    def _convert_messages_to_gemini_format(self, messages: list[ChatMessage | dict]) -> list[dict]:
        """将标准聊天消息转换为Gemini API要求的格式
        
        Gemini API要求：
        - 只接受user和model两种角色
        - 内容必须放在parts数组中
        - 支持多模态内容（文本/图像）
        
        Args:
            messages: 原始消息列表（ChatMessage对象或字典）
            
        Returns:
            转换后的消息列表，符合Gemini API格式要求
        """
        converted = []
        for msg in messages:
            # 统一转换为字典格式
            if isinstance(msg, ChatMessage):
                msg = msg.dict()  # 使用ChatMessage的dict方法
            
            # 应用角色转换（确保最终只有user或model两种角色）
            original_role = msg["role"]
            role = self.custom_role_conversions.get(original_role, "user")
            if role not in ["user", "model"]:
                role = "user"  # 安全回退
            
            # 处理消息内容（支持多模态）
            content = msg["content"]
            parts = []
            
            if isinstance(content, list):
                # 多模态内容处理（文本+图像）
                for part in content:
                    if isinstance(part, dict):
                        if "text" in part:
                            parts.append({"text": part["text"]})
                        elif "image_url" in part:
                            # 图像URL处理（需确保URL可访问）
                            parts.append({"image_url": part["image_url"]})
                    else:
                        parts.append({"text": str(part)})
            else:
                # 纯文本内容
                parts = [{"text": str(content)}]
            
            # 构建符合Gemini格式的消息
            converted.append({
                "role": role,
                "parts": parts  # Gemini使用parts数组存放内容
            })
        
        return converted

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """非流式消息生成
        
        Args:
            messages: 聊天消息历史
            stop_sequences: 停止生成序列
            response_format: 响应格式要求
            tools_to_call_from: 可用工具列表
            **kwargs: 其他生成参数
            
        Returns:
            ChatMessage: 包含模型响应的消息对象
        """
        # 获取模型实例
        model = self.client.GenerativeModel(self.model_id)
        
        # 转换消息格式
        gemini_messages = self._convert_messages_to_gemini_format(messages)
        
        # 构建生成配置
        generation_config = {
            "stop_sequences": stop_sequences or [],  # 确保不为None
            **(response_format or {}),  # 响应格式
            **kwargs,  # 其他参数（如temperature等）
        }
        
        # 应用速率限制
        self._apply_rate_limit()
        
        # 调用API生成内容
        response = model.generate_content(
            contents=gemini_messages,
            generation_config=generation_config,
        )
        # 构造返回消息（将model角色转换回assistant）
        return ChatMessage(
            role="assistant",  # 统一转换为assistant角色
            content=response.text,
            raw=response,  # 保留原始响应
            token_usage=TokenUsage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            ) if hasattr(response, "usage_metadata") else None
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta, None, None]:
        """流式消息生成
        
        Args:
            参数同generate方法
            
        Yields:
            ChatMessageStreamDelta: 流式响应增量
        """
        model = self.client.GenerativeModel(self.model_id)
        gemini_messages = self._convert_messages_to_gemini_format(messages)
        
        generation_config = {
            "stop_sequences": stop_sequences or [],
            **(response_format or {}),
            **kwargs,
        }
        
        self._apply_rate_limit()
        
        # 流式调用
        response = model.generate_content(
            contents=gemini_messages,
            stream=True,  # 启用流式
            generation_config=generation_config,
        )
        
        # 逐块返回响应
        for chunk in response:
            yield ChatMessageStreamDelta(
                content=chunk.text,
                token_usage=TokenUsage(
                    input_tokens=getattr(chunk, "prompt_token_count", 0),
                    output_tokens=getattr(chunk, "candidates_token_count", 0),
                ) if hasattr(chunk, "prompt_token_count") else None,
            )


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "ApiModel",
    "OpenAIServerModel",
    "GeminiServerModel",
    "ChatMessage",
]