#!/usr/bin/env python3
"""
大模型调用示例 - run_model.py

基于 OpenAI 兼容 API 的大模型调用实现，支持流式和非流式调用。
"""

import os
import sys
from typing import List

from dotenv import load_dotenv

from antagents import (
    BuiltinToolEventStreamDelta,
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    OpenAIServerModel,
    TokenUsage,
    agglomerate_stream_deltas,
)


def safe_print(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print text safely on Windows terminals with legacy encodings.

    Some responses contain characters that cannot be encoded by GBK/CP936.
    When that happens, fall back to a replacement-based write so the example
    keeps running instead of crashing in the middle of a stream.
    """

    text = sep.join(str(arg) for arg in args)
    output = text + end
    try:
        print(text, sep=sep, end=end, flush=flush)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_output = output.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_output)
        if flush:
            sys.stdout.flush()


def print_builtin_tool_events(events: List[BuiltinToolEventStreamDelta], prefix: str = "") -> None:
    """Pretty-print structured built-in tool events from the responses API."""
    if not events:
        return
    safe_print(f"\n{prefix}[builtin] 内建工具事件:")
    for event in events:
        item_suffix = f" (id={event.item_id})" if event.item_id else ""
        safe_print(f"{prefix}- {event.tool_type}: {event.status}{item_suffix}")


def create_example_messages() -> List[ChatMessage]:
    """创建示例消息"""
    return [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="你是一个有帮助的AI助手，请用中文回答用户的问题，回答要简洁明了。",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="请解释一下人工智能的基本概念和应用领域。",
        ),
    ]


def print_messages_detailed(messages: List[ChatMessage]) -> None:
    """打印消息详情。"""
    safe_print("[input] 输入的 Prompt 详情:")
    safe_print("=" * 60)

    role_labels = {
        MessageRole.SYSTEM: "[SYSTEM]",
        MessageRole.USER: "[USER]",
        MessageRole.ASSISTANT: "[ASSISTANT]",
        MessageRole.TOOL_CALL: "[TOOL_CALL]",
        MessageRole.TOOL_RESPONSE: "[TOOL_RESPONSE]",
    }

    for i, msg in enumerate(messages, 1):
        role_label = role_labels.get(msg.role, "[MESSAGE]")
        safe_print(f"{role_label} 消息 {i} [{msg.role.value.upper()}]:")

        if isinstance(msg.content, list):
            safe_print(f"   内容类型: 多模态 ({len(msg.content)} 个部分)")
            for j, part in enumerate(msg.content, 1):
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        content_preview = text[:100] + "..." if len(text) > 100 else text
                        safe_print(f"      {j}. 文本: {content_preview}")
                    elif part.get("type") in ["image", "image_url"]:
                        safe_print(f"      {j}. 图像: [图像内容]")
        else:
            text = str(msg.content)
            content_preview = text[:100] + "..." if text and len(text) > 100 else msg.content
            safe_print(f"   内容: {content_preview}")

        if msg.tool_calls:
            safe_print(f"   工具调用: {len(msg.tool_calls)} 个")
            for tool_call in msg.tool_calls:
                if hasattr(tool_call, "function"):
                    safe_print(f"      - {tool_call.function.name}")

        if msg.token_usage:
            safe_print(f"   Token 使用: 输入={msg.token_usage.input_tokens}, 输出={msg.token_usage.output_tokens}")

        if i < len(messages):
            safe_print("   " + "-" * 40)


def non_streaming_example(model: OpenAIServerModel, messages: List[ChatMessage]) -> bool:
    """非流式调用示例。"""
    safe_print("\n[run] 开始非流式调用...")
    safe_print("=" * 60)

    try:
        response = model.generate(messages=messages)

        safe_print("[ok] 非流式调用成功")
        safe_print("\n[reply] 模型回复:")
        safe_print("-" * 40)
        safe_print(response.content)
        safe_print("-" * 40)

        if response.token_usage:
            safe_print("\n[token] Token 使用统计:")
            safe_print(f"   输入 Token: {response.token_usage.input_tokens}")
            safe_print(f"   输出 Token: {response.token_usage.output_tokens}")
            safe_print(f"   总 Token: {response.token_usage.input_tokens + response.token_usage.output_tokens}")

        if response.tool_calls:
            safe_print("\n[tools] 工具调用:")
            for tool_call in response.tool_calls:
                if hasattr(tool_call, "function"):
                    safe_print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")

    except Exception as e:
        safe_print(f"[error] 非流式调用过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def streaming_example(model: OpenAIServerModel, messages: List[ChatMessage]) -> bool:
    """流式调用示例。"""
    safe_print("\n[run] 开始流式调用...")
    safe_print("=" * 60)

    try:
        safe_print("[stream] 模型回复:")
        safe_print("-" * 40)

        final_token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        stream_deltas: List[ChatMessageStreamDelta] = []

        for delta in model.generate_stream(messages=messages):
            stream_deltas.append(delta)
            if delta.content:
                safe_print(delta.content, end="", flush=True)

            if delta.builtin_tool_events:
                print_builtin_tool_events(delta.builtin_tool_events, prefix="   ")

            if delta.token_usage:
                final_token_usage.input_tokens += delta.token_usage.input_tokens
                final_token_usage.output_tokens += delta.token_usage.output_tokens

        safe_print("\n" + "-" * 40)

        safe_print("\n[token] Token 使用统计:")
        safe_print(f"   输入 Token: {final_token_usage.input_tokens}")
        safe_print(f"   输出 Token: {final_token_usage.output_tokens}")
        safe_print(f"   总 Token: {final_token_usage.input_tokens + final_token_usage.output_tokens}")

        aggregated = agglomerate_stream_deltas(stream_deltas)
        if aggregated.tool_calls:
            safe_print("\n[tools] 工具调用:")
            for tool_call in aggregated.tool_calls:
                safe_print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")

    except Exception as e:
        safe_print(f"[error] 流式调用过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def interactive_chat(model: OpenAIServerModel) -> None:
    """交互式聊天示例。"""
    safe_print("\n[chat] 进入交互式聊天模式 (输入 'quit' 退出)")
    safe_print("=" * 60)

    system_message = ChatMessage(
        role=MessageRole.SYSTEM,
        content="你是一个有帮助的AI助手，请用中文进行友好、专业的对话。",
    )
    messages = [system_message]

    while True:
        try:
            user_input = input("\n您: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                safe_print("退出交互模式")
                break

            if not user_input:
                continue

            messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            safe_print("\nAI: ", end="", flush=True)

            response_content = ""
            token_usage = TokenUsage(input_tokens=0, output_tokens=0)
            stream_deltas: List[ChatMessageStreamDelta] = []

            for delta in model.generate_stream(messages=messages):
                stream_deltas.append(delta)
                if delta.content:
                    safe_print(delta.content, end="", flush=True)
                    response_content += delta.content

                if delta.builtin_tool_events:
                    print_builtin_tool_events(delta.builtin_tool_events, prefix="   ")

                if delta.token_usage:
                    token_usage.input_tokens += delta.token_usage.input_tokens
                    token_usage.output_tokens += delta.token_usage.output_tokens

            aggregated = agglomerate_stream_deltas(stream_deltas)
            if aggregated.tool_calls:
                safe_print("\n   [tools] 工具调用:")
                for tool_call in aggregated.tool_calls:
                    safe_print(f"   - {tool_call.function.name}: {tool_call.function.arguments}")

            if response_content:
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_content,
                        token_usage=token_usage,
                    )
                )

            safe_print(f"\n   [token] 本次调用: 输入={token_usage.input_tokens}, 输出={token_usage.output_tokens}")

        except KeyboardInterrupt:
            safe_print("\n用户中断，退出交互模式")
            break
        except EOFError:
            safe_print("\n[info] 检测到非交互输入结束，退出交互式聊天示例")
            break
        except Exception as e:
            safe_print(f"\n[error] 聊天过程中出现错误: {e}")
            import traceback

            traceback.print_exc()
            break


def main() -> None:
    """主函数。"""
    safe_print("模型调用示例程序")
    safe_print("=" * 60)

    try:
        api_mode = os.getenv("OPENAI_API_MODE", "auto")
        model = OpenAIServerModel(
            model_id=os.getenv("DEEPSEEK_MODEL_ID"),
            api_base=os.getenv("DEEPSEEK_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_mode=api_mode,
        )
        safe_print(f"[ok] 模型初始化成功: {model.model_id} (api_mode={api_mode})")

    except Exception as e:
        safe_print(f"[error] 模型初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return

    messages = create_example_messages()
    print_messages_detailed(messages)

    success = non_streaming_example(model, messages)
    if not success:
        safe_print("[warn] 非流式调用失败，跳过后续示例")
        return

    success = streaming_example(model, messages)
    if not success:
        safe_print("[warn] 流式调用失败")
        return

    if sys.stdin.isatty():
        try:
            interactive_chat(model)
        except Exception as e:
            safe_print(f"[error] 交互式聊天失败: {e}")
            import traceback

            traceback.print_exc()
    else:
        safe_print("[info] 检测到非交互终端，跳过交互式聊天示例")

    safe_print("\n程序执行完成")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
