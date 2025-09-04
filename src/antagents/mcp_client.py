#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any

from antagents.tools import Tool


__all__ = ["MCPClient"]

if TYPE_CHECKING:
    from mcpadapt.core import StdioServerParameters


class MCPClient:
    """管理与MCP服务器的连接，并将其工具提供给antagents使用。

    注意：工具只能在通过`connect()`方法启动连接后访问，该方法在初始化时完成。如果不使用上下文管理器，
    我们强烈建议使用"try ... finally"来确保连接被正确清理。

    参数:
        server_parameters (StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]]):
            连接MCP服务器的配置参数。如果要同时连接多个MCP，可以是列表。

            - `mcp.StdioServerParameters`的实例，用于通过子进程的标准输入/输出连接Stdio MCP服务器。

            - 一个`dict`，至少包含：
              - "url": 服务器的URL。
              - "transport": 使用的传输协议，可选：
                - "streamable-http": （推荐）可流式HTTP传输。
                - "sse": 传统的HTTP+SSE传输（已弃用）。
              如果省略"transport"，则默认为传统的"sse"传输（会发出弃用警告）。

            <Deprecated version="1.17.0">
            HTTP+SSE传输已弃用，未来行为将默认为可流式HTTP传输。
            请显式传递"transport"键。
            </Deprecated>

        adapter_kwargs (dict[str, Any], optional):
            直接传递给`MCPAdapt`的额外关键字参数。

    示例:
        ```python
        # 完全托管的上下文管理器 + stdio
        with MCPClient(...) as tools:
            # 工具现在可用

        # 上下文管理器 + 可流式HTTP传输:
        with MCPClient({"url": "http://localhost:8000/mcp", "transport": "streamable-http"}) as tools:
            # 工具现在可用

        # 通过mcp_client对象手动管理连接:
        try:
            mcp_client = MCPClient(...)
            tools = mcp_client.get_tools()

            # 在此处使用工具。
        finally:
            mcp_client.disconnect()
        ```
    """

    def __init__(
        self,
        server_parameters: "StdioServerParameters" | dict[str, Any] | list["StdioServerParameters" | dict[str, Any]],
        adapter_kwargs: dict[str, Any] | None = None,
    ):
        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.antagents_adapter import antagentsAdapter
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install 'mcp' extra to use MCPClient: `pip install 'antagents[mcp]'`")
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
        adapter_kwargs = adapter_kwargs or {}
        self._adapter = MCPAdapt(server_parameters, antagentsAdapter(), **adapter_kwargs)
        self._tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        """连接到MCP服务器并初始化工具。"""
        self._tools: list[Tool] = self._adapter.__enter__()

    def disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ):
        """断开与MCP服务器的连接"""
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)

    def get_tools(self) -> list[Tool]:
        """从MCP服务器获取可用的antagents工具。

        注意：目前，此方法总是返回会话创建时可用的工具，
        但在未来版本中，它将返回调用时MCP服务器提供的任何新工具。

        异常:
            ValueError: 如果MCP服务器工具为None（通常表示服务器未启动）。

        返回:
            list[Tool]: 从MCP服务器可用的antagents工具。
        """
        if self._tools is None:
            raise ValueError(
                "Couldn't retrieve tools from MCP server, run `mcp_client.connect()` first before accessing `tools`"
            )
        return self._tools

    def __enter__(self) -> list[Tool]:
        """连接到MCP服务器并直接返回工具。

        注意：由于初始化时的`.connect`，此时mcp_client已经连接。
        """
        return self._tools

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        """断开与MCP服务器的连接。"""
        self.disconnect(exc_type, exc_value, exc_traceback)