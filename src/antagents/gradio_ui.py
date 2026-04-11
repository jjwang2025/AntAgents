#!/usr/bin/env python
# coding=utf-8

import os
import re
import shutil
import html
import importlib.resources
from pathlib import Path
from typing import Generator

from antagents.agent_types import AgentAudio, AgentImage, AgentText
from antagents.agents import MultiStepAgent, PlanningStep
from antagents.memory import ActionStep, FinalAnswerStep
from antagents.models import BuiltinToolEventStreamDelta, ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from antagents.utils import _is_package_available, decode_unicode_escapes


CHAT_UI_CSS = """
html, body {
  overflow: hidden;
  height: 100%;
}

body {
  margin: 0;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .app,
#chat-main {
  overflow: hidden !important;
}

#chat-main {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: calc(100vh - 24px);
  min-height: 0;
}

#chatbot-panel {
  flex: 1 1 auto;
  min-height: 0;
  height: 100% !important;
  max-height: none !important;
  overflow: hidden !important;
}

#chat-composer-panel {
  position: sticky;
  bottom: 0;
  z-index: 20;
  border: none !important;
  border-radius: 0;
  background: transparent !important;
  box-shadow: none !important;
  padding: 8px 12px 12px 12px;
  overflow: visible !important;
}

#chat-composer-row {
  align-items: center !important;
  gap: 10px !important;
  overflow: visible !important;
}

#chat-composer-panel > div,
#chat-composer-panel .gr-group {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  overflow: visible !important;
}

#chat-composer-input {
  width: 100% !important;
  border: 1px solid rgba(148, 163, 184, 0.35) !important;
  border-radius: 18px !important;
  background: #ffffff !important;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
  min-height: 44px !important;
  padding: 2px 12px !important;
  overflow: visible !important;
}

#chat-composer-input textarea {
  border: none !important;
  box-shadow: none !important;
  background: #ffffff !important;
  font-size: 15px !important;
  line-height: 1.55 !important;
  min-height: 30px !important;
  max-height: 180px !important;
  overflow-y: hidden !important;
  resize: none !important;
  padding: 10px 2px !important;
}

#chat-composer-input:focus-within {
  border-color: rgba(37, 99, 235, 0.65) !important;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
}

#chat-composer-input textarea:focus {
  border: none !important;
  box-shadow: none !important;
}

#chat-send-btn button {
  border-radius: 999px !important;
  min-width: 54px !important;
  width: 54px !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  height: 42px !important;
  font-weight: 600 !important;
}

#composer-inline-hint {
  display: flex;
  justify-content: flex-end;
  margin: 0 4px 8px 4px;
}

#composer-inline-hint span {
  padding: 6px 10px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.96);
  color: #64748b;
  font-size: 12px;
  line-height: 1.2;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
  white-space: nowrap;
}

#chat-run-status {
  margin: 0 4px 8px 4px;
}

#chat-run-status .chat-run-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 999px;
  background: rgba(239, 246, 255, 0.96);
  color: #1d4ed8;
  font-size: 13px;
  line-height: 1.2;
}

#chat-run-status .chat-run-status-pill::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #3b82f6;
  box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.45);
  animation: antagents-status-pulse 1.6s infinite;
}

@keyframes antagents-status-pulse {
  0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.45); }
  70% { box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
  100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
}
"""


CHAT_UI_HEAD = """
<script>
(() => {
  function autoResize(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px';
  }

  function bindComposerEnter() {
    const textarea = document.querySelector('#chat-composer-input textarea');
    if (!textarea || textarea.dataset.antagentsBound === '1') return;

    textarea.dataset.antagentsBound = '1';
    autoResize(textarea);

    textarea.addEventListener('input', () => autoResize(textarea));
  }

  document.addEventListener('DOMContentLoaded', bindComposerEnter);
  const observer = new MutationObserver(bindComposerEnter);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """获取步骤日志的脚注字符串，包含持续时间和令牌信息"""
    step_footnote = f"{step_name}"
    duration = round(float(step_log.timing.duration), 2) if step_log.timing.duration else None
    if step_log.token_usage is not None:
        step_footnote += f" | 输入 Token: {step_log.token_usage.input_tokens:,} | 输出 Token: {step_log.token_usage.output_tokens:,}"
        if duration and duration > 0:
            tokens_per_second = step_log.token_usage.output_tokens / duration
            step_footnote += f" | 生成速度: {tokens_per_second:.1f} tok/s"
    step_footnote += f" | 耗时: {duration}s" if duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def _format_builtin_event(event: BuiltinToolEventStreamDelta) -> str:
    item_suffix = f"\n- ID: `{event.item_id}`" if event.item_id else ""
    raw_suffix = f"\n- 原始事件: `{event.raw_type}`" if event.raw_type else ""
    return (
        f"### 内建工具事件\n"
        f"- 工具类型: `{event.tool_type}`\n"
        f"- 状态: `{event.status}`"
        f"{item_suffix}"
        f"{raw_suffix}"
    )


def _simple_markdown_to_html(text: str) -> str:
    """Render a small Markdown subset for Gradio card bodies.

    This keeps planning headers, lists, code fences and inline emphasis readable
    inside HTML cards without pulling in an extra markdown runtime dependency.
    """
    if not text.strip():
        return ""

    blocks: list[str] = []
    current_list: list[str] = []
    in_code_block = False
    code_lines: list[str] = []

    def flush_list() -> None:
        nonlocal current_list
        if current_list:
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in current_list) + "</ul>")
            current_list = []

    def render_inline(value: str) -> str:
        escaped = html.escape(value)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        return escaped

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code_block = False
            else:
                flush_list()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw_line)
            continue

        if not stripped:
            flush_list()
            continue

        if stripped.startswith("### "):
            flush_list()
            blocks.append(f"<h3>{render_inline(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            flush_list()
            blocks.append(f"<h2>{render_inline(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            flush_list()
            blocks.append(f"<h1>{render_inline(stripped[2:])}</h1>")
            continue
        if stripped.startswith(("- ", "* ")):
            current_list.append(render_inline(stripped[2:]))
            continue

        flush_list()
        blocks.append(f"<p>{render_inline(stripped)}</p>")

    flush_list()
    if in_code_block:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    return "".join(blocks)


def _sanitize_ui_text(text: str) -> str:
    """Remove runtime/system artifacts that should never surface in the Gradio chat UI."""
    cleaned = text
    cleaned = re.sub(r"<system-reminder\b[^>]*>.*?(?:</system-reminder>|$)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</system-reminder>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Your operational mode has changed from .*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"You are no longer in read-only mode\.", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed\.",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _render_card(title: str, body: str, accent: str, subtitle: str | None = None, body_is_html: bool = False) -> str:
    subtitle_html = f"<div style='font-size:12px;color:#94a3b8;margin-top:2px;'>{subtitle}</div>" if subtitle else ""
    rendered_body = body if body_is_html else _simple_markdown_to_html(_sanitize_ui_text(body))
    return (
        f"<div style='width:680px; max-width:100%; box-sizing:border-box; border:1px solid {accent}33; border-left:4px solid {accent}; border-radius:16px; "
        "padding:14px 16px; background:rgba(255,255,255,0.92); box-shadow:0 6px 16px rgba(15,23,42,0.05);'>"
        f"<div style='font-size:12px; letter-spacing:0.04em; text-transform:uppercase; color:{accent}; font-weight:700;'>{title}</div>"
        f"{subtitle_html}"
        f"<div style='margin-top:10px;'>{rendered_body}</div>"
        "</div>"
    )


def _highlight_final_answer(text: str) -> str:
    return _render_card("Final Answer", text, "#0ea5e9", subtitle="最终输出")


def _format_tool_list_markdown(items: list[tuple[str, str]]) -> str:
    if not items:
        return "- 无"
    return "\n".join([f"- `{name}`: {description}" if description else f"- `{name}`" for name, description in items])


def _render_streaming_reply(text: str, body_is_html: bool = False) -> str:
    if body_is_html:
        return text
    return _sanitize_ui_text(text)


def _format_streaming_markdown(text: str) -> str:
    text = _sanitize_ui_text(text)
    if not text.strip():
        return ""
    return _render_streaming_reply(text)


def _clean_model_output(model_output: str) -> str:
    """
    通过移除尾部标签和多余的引号来清理模型输出。

    参数:
        model_output (`str`): 原始模型输出。

    返回:
        `str`: 清理后的模型输出。
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # 移除任何尾部的<end_code>和多余的引号，处理多种可能的格式
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # 处理```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # 处理<end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # 处理```\n<end_code>
    return _sanitize_ui_text(model_output.strip())


def _format_code_content(content: str) -> str:
    """
    如果内容尚未格式化，则将其格式化为Python代码块。

    参数:
        content (`str`): 要格式化的代码内容。

    返回:
        `str`: 格式化为Python代码块的内容。
    """
    content = content.strip()
    # 移除现有的代码块和end_code标签
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # 如果尚未添加Python代码块格式，则添加
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    """
    处理一个[`ActionStep`]并生成适当的Gradio ChatMessage对象。

    参数:
        step_log ([`ActionStep`]): 要处理的ActionStep。
        skip_model_outputs (`bool`): 是否跳过模型输出。

    生成:
        `gradio.ChatMessage`: 表示动作步骤的Gradio ChatMessages。
    """
    import gradio as gr

    # 输出步骤编号
    step_number = f"执行步骤 {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card(step_number, "正在执行当前步骤。", "#8b5cf6", subtitle="Action Step"),
            metadata={"title": step_number, "status": "done", "id": f"action-step-{step_log.step_number}"},
        )

    # 首先生成LLM的思考/推理
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card("模型输出", model_output, "#6366f1", subtitle="Reasoning / Draft"),
            metadata={"status": "done"},
        )

    # 对于工具调用，创建一个父消息
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # 根据类型处理参数
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # 如果需要，格式化代码内容
        if used_code:
            content = _render_card(
                "工具调用",
                f"`{first_tool_call.name}`\n\n{_format_code_content(content)}",
                "#14b8a6",
                subtitle="Python 执行",
            )
        else:
            content = _render_card(
                "工具调用",
                f"**{first_tool_call.name}**\n\n`{content}`",
                "#14b8a6",
                subtitle="Tool Call",
            )

        # 创建工具调用消息
        parent_message_tool = gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "title": f"工具调用: {first_tool_call.name}",
                "status": "done",
                "id": f"tool-call-{step_log.step_number}-{first_tool_call.name}",
            },
        )
        yield parent_message_tool

    # 如果存在执行日志，则显示
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = step_log.observations.strip()
        if log_content:
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content=_render_card("执行日志", f"```bash\n{log_content}\n```", "#f59e0b", subtitle="Observation"),
                metadata={"title": "执行日志", "status": "done", "id": f"execution-log-{step_log.step_number}"},
            )

    # 显示观察中的任何图像
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                metadata={"title": "输出图片", "status": "done", "id": f"output-image-{step_log.step_number}"},
            )

    # 处理错误
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card("错误", str(step_log.error), "#ef4444", subtitle="Execution Error"),
            metadata={"title": "错误", "status": "done", "id": f"action-error-{step_log.step_number}"},
        )

    # 添加步骤脚注和分隔符
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, step_number),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    """
    处理一个[`PlanningStep`]并生成适当的gradio.ChatMessage对象。

    参数:
        step_log ([`PlanningStep`]): 要处理的PlanningStep。

    生成:
        `gradio.ChatMessage`: 表示规划步骤的Gradio ChatMessages。
    """
    import gradio as gr

    if not skip_model_outputs:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card("规划步骤", "已生成新的任务计划。", "#8b5cf6", subtitle="Planning"),
            metadata={"title": "规划步骤", "status": "done", "id": f"planning-step-{step_log.step_number}"},
        )
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card("计划内容", step_log.plan, "#8b5cf6", subtitle="Plan Body"),
            metadata={"title": "计划内容", "status": "done", "id": f"plan-body-{step_log.step_number}"},
        )
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, "规划步骤"),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    """
    处理一个[`FinalAnswerStep`]并生成适当的gradio.ChatMessage对象。

    参数:
        step_log ([`FinalAnswerStep`]): 要处理的FinalAnswerStep。

    生成:
        `gradio.ChatMessage`: 表示最终答案的Gradio ChatMessages。
    """
    import gradio as gr

    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_highlight_final_answer(final_answer.to_string()),
            metadata={"status": "done", "id": "final-answer"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done", "id": "final-answer"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"status": "done", "id": "final-answer"},
        )
    else:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_highlight_final_answer(str(final_answer)),
            metadata={"status": "done", "id": "final-answer"},
        )


def _process_builtin_tool_events(events: list[BuiltinToolEventStreamDelta]) -> Generator:
    """将结构化的 responses 内建工具事件渲染为独立聊天消息。"""
    import gradio as gr

    for event in events:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=_render_card("内建工具事件", _format_builtin_event(event), "#f59e0b", subtitle="Responses Built-in Tool"),
            metadata={"title": f"内建工具: {event.tool_type}", "status": "done", "id": f"builtin-{event.tool_type}-{event.item_id or event.index}"},
        )


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    """从智能体步骤中提取Gradio ChatMessage对象，并正确嵌套。

    参数:
        step_log: 要显示为gr.ChatMessage对象的步骤日志。
        skip_model_outputs: 如果为True，则在创建gr.ChatMessage对象时跳过模型输出：
            例如，当流式模型输出已经显示时使用。
    """
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'antagents[gradio]'`"
        )
    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
) -> Generator:
    """使用给定任务运行智能体，并将智能体的消息流式传输为gradio ChatMessages。"""

    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'antagents[gradio]'`"
        )
    accumulated_events: list[ChatMessageStreamDelta] = []
    for event in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
            for message in pull_messages_from_step(
                event,
                # 如果正在流式传输模型输出，无需重复显示
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            if event.builtin_tool_events:
                for message in _process_builtin_tool_events(event.builtin_tool_events):
                    yield message
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            text = re.sub(r"\n\[(?:[a-z_]+_call|mcp_list_tools)\] [a-z_]+(?: \([^)]+\))?", "", text)
            # 解码 Unicode 转义字符
            decoded_text = decode_unicode_escapes(text)
            yield decoded_text


class GradioUI:
    """
    用于与[`MultiStepAgent`]交互的Gradio界面。

    该类提供了一个Web界面，可以实时与智能体交互，允许用户提交提示、上传文件并以聊天格式接收响应。
    如果需要，可以在每次交互开始时重置智能体的记忆。
    它支持文件上传，文件将保存到指定的文件夹。
    它使用[`gradio.Chatbot`]组件来显示对话历史。
    该类需要安装`gradio`额外组件：`antagents[gradio]`。

    参数:
        agent ([`MultiStepAgent`]): 要交互的智能体。
        file_upload_folder (`str`, *可选*): 上传文件将保存到的文件夹。
            如果未提供，则禁用文件上传。
        reset_agent_memory (`bool`, *可选*, 默认为`False`): 是否在每次交互开始时重置智能体的记忆。
            如果为`True`，智能体将不会记住之前的交互。

    抛出:
        ModuleNotFoundError: 如果未安装`gradio`额外组件。

    示例:
        ```python
        from antagents import ToolCallingAgent, GradioUI, OpenAIServerModel

        model = OpenAIServerModel(
            model_id=os.getenv("DEEPSEEK_MODEL_ID"),
            api_base=os.getenv("DEEPSEEK_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        agent = ToolCallingAgent(tools=[], model=model)
        gradio_ui = GradioUI(agent, file_upload_folder="uploads", reset_agent_memory=True)
        gradio_ui.launch()
        ```
    """

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None, reset_agent_memory: bool = False):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'antagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = Path(file_upload_folder) if file_upload_folder is not None else None
        self.reset_agent_memory = reset_agent_memory
        self.name = getattr(agent, "display_name", None) or getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)

    def _available_tools_for_display(self) -> list[tuple[str, str]]:
        tools = getattr(self.agent, "tools", {}) or {}
        return [(name, getattr(tool, "description", "")) for name, tool in tools.items()]

    def _selected_tools_for_display(self) -> list[tuple[str, str]]:
        if hasattr(self.agent, "tools_and_managed_agents"):
            return [
                (getattr(tool, "name", str(tool)), getattr(tool, "description", ""))
                for tool in self.agent.tools_and_managed_agents
            ]
        return self._available_tools_for_display()

    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        def _message_metadata(message):
            if isinstance(message, dict):
                return message.get("metadata", {}) or {}
            return getattr(message, "metadata", {}) or {}

        def _set_message_content(message, content):
            if isinstance(message, dict):
                message["content"] = content
            else:
                message.content = content

        # 从模板智能体获取智能体类型
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages = [msg for msg in messages if _message_metadata(msg).get("id") != "streaming-reply"]
            streaming_index = None

            for msg in stream_to_gradio(
                session_state["agent"], task=prompt, reset_agent_memory=self.reset_agent_memory
            ):
                if isinstance(msg, gr.ChatMessage):
                    if streaming_index is not None and streaming_index < len(messages):
                        messages.pop(streaming_index)
                        streaming_index = None
                    messages.append(msg)
                elif isinstance(msg, str):  # 那么它只是一个完成增量
                    if msg.strip() in {"...", "…", "正在思考...", "正在思考…"}:
                        continue
                    # Keep Markdown formatting for the streaming card instead of escaping it into plain text.
                    msg = _format_streaming_markdown(msg)
                    if not msg.strip():
                        continue
                    if streaming_index is not None and streaming_index < len(messages):
                        _set_message_content(messages[streaming_index], msg)
                    else:
                        messages.append(
                            gr.ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=msg,
                            )
                        )
                        streaming_index = len(messages) - 1
                yield messages

            yield messages
        except Exception as e:
            yield messages
            raise gr.Error(f"Error in interaction: {str(e)}")

    def submit_user_message(self, text_input, messages, file_uploads_log):
        """Append the user message immediately, clear the composer, and disable submit while running."""
        import gradio as gr

        prompt = text_input.strip()
        if len(file_uploads_log) > 0:
            prompt += (
                f"\n你还提供了这些文件，可按需使用：{file_uploads_log}"
                if prompt
                else f"你还提供了这些文件，可按需使用：{file_uploads_log}"
            )

        if not prompt:
            return messages, "", gr.Button(interactive=True), gr.Textbox(interactive=True), "", gr.HTML(visible=False)

        updated_messages = list(messages)
        updated_messages.append(gr.ChatMessage(role="user", content=prompt, metadata={"status": "done"}))
        return (
            updated_messages,
            "",
            gr.Button(interactive=False),
            gr.Textbox(interactive=False),
            prompt,
            gr.HTML("<div class='chat-run-status-pill'>正在调用工具并生成结果...</div>", visible=True),
        )

    def restore_input_state(self):
        import gradio as gr

        return (
            gr.Textbox(interactive=True, placeholder="给智能体下达任务，Enter 发送，Shift+Enter 换行"),
            gr.Button(interactive=True),
            gr.HTML(visible=False),
        )

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        上传文件并将其添加到会话状态的上传文件列表中。

        文件将保存到`self.file_upload_folder`文件夹。
        如果文件类型不允许，则返回指示不允许的文件类型的消息。

        参数:
            file (`gradio.File`): 上传的文件。
            file_uploads_log (`list`): 记录上传文件的列表。
            allowed_file_types (`list`, *可选*): 允许的文件扩展名列表。默认为[".pdf", ".docx", ".txt"]。
        """
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # 清理文件名
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # 将任何非字母数字、非短横线或非点字符替换为下划线

        # 将上传的文件保存到指定文件夹
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def launch(self, share: bool = True, **kwargs):
        """
        启动带有智能体界面的Gradio应用。

        参数:
            share (`bool`, 默认为`True`): 是否公开分享应用。
            **kwargs: 传递给Gradio启动方法的额外关键字参数。
        """
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        import gradio as gr

        agent_avatar_path = str(
            importlib.resources.files("antagents.assets").joinpath("agent_avatar.svg")
        )

        with gr.Blocks(theme="ocean", fill_height=True, css=CHAT_UI_CSS, head=CHAT_UI_HEAD) as demo:
            # 添加会话状态以存储会话特定数据
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> 在这里可以直接体验工具调用、规划步骤、流式输出，以及 responses 内建工具事件展示。"
                    + (f"\n\n**智能体描述**\n{self.description}" if self.description else "")
                )

                gr.Markdown(
                    "**如何使用多智能体编排**\n"
                    "- AntAgents 总控台负责理解任务、拆分步骤，并决定何时调用家族成员。\n"
                    "- 侦察蚁负责网页搜索和页面访问，适合处理需要查资料、比对网页、汇总来源的问题。\n"
                    "- 适合直接输入较完整的任务目标，而不是只输入一个关键词。\n\n"
                    "**推荐提问方式**\n"
                    "- 先说明目标，再说明你希望它如何组织信息。\n"
                    "- 如果需要联网检索，直接写清楚“请先搜索并给出结论”。\n"
                    "- 如果需要多步处理，明确写出“先查资料，再比较，最后总结”。\n\n"
                    "**示例任务**\n"
                    "- 请先搜索 OpenAI 最新的 GPT-5 公开说明，再总结它和 GPT-4o 的主要差异。\n"
                    "- 请搜索 2024 年中国 GDP 增速相关数据，说明如果保持该增速，GDP 大约多少年翻倍。\n"
                    "- 请先搜索某个产品的官网和文档页面，再整理一个简短的功能对比表。"
                )

                gr.Markdown("**可用工具**\n" + _format_tool_list_markdown(self._available_tools_for_display()))
                gr.Markdown("**当前启用工具**\n" + _format_tool_list_markdown(self._selected_tools_for_display()))

                # 如果提供了上传文件夹，则启用上传功能
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="上传文件")
                    upload_status = gr.Textbox(label="上传状态", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                gr.HTML(
                    "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/antagents'><b>antagents</b></a></center></h4>"
                )

            with gr.Column(scale=1, elem_id="chat-main"):
                chatbot = gr.Chatbot(
                    label="智能体轨迹",
                    type="messages",
                    avatar_images=(None, agent_avatar_path),
                    container=False,
                    resizeable=False,
                    scale=1,
                    bubble_full_width=False,
                    show_copy_button=True,
                    autoscroll=True,
                    height="100%",
                    elem_id="chatbot-panel",
                    latex_delimiters=[
                        {"left": r"$$", "right": r"$$", "display": True},
                        {"left": r"$", "right": r"$", "display": False},
                        {"left": r"\[", "right": r"\]", "display": True},
                        {"left": r"\(", "right": r"\)", "display": False},
                    ],
                )

                with gr.Group(elem_id="chat-composer-panel"):
                    run_status = gr.HTML(visible=False, elem_id="chat-run-status")
                    gr.HTML("<div id='composer-inline-hint'><span>Enter 发送 · Shift+Enter 换行</span></div>")
                    with gr.Row(equal_height=False, elem_id="chat-composer-row"):
                        text_input = gr.Textbox(
                            lines=1,
                            max_lines=8,
                            show_label=False,
                            container=False,
                            placeholder="给智能体下达任务，Enter 发送，Shift+Enter 换行",
                            autofocus=True,
                            elem_id="chat-composer-input",
                            scale=20,
                        )
                        submit_btn = gr.Button("发送", variant="primary", elem_id="chat-send-btn", scale=1)

            # 设置事件处理器
            submit_btn.click(
                self.submit_user_message,
                [text_input, chatbot, file_uploads_log],
                [chatbot, text_input, submit_btn, text_input, stored_messages, run_status],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                self.restore_input_state,
                None,
                [text_input, submit_btn, run_status],
            )

            text_input.submit(
                self.submit_user_message,
                [text_input, chatbot, file_uploads_log],
                [chatbot, text_input, submit_btn, text_input, stored_messages, run_status],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                self.restore_input_state,
                None,
                [text_input, submit_btn, run_status],
            )

            chatbot.clear(self.agent.memory.reset)
        return demo


__all__ = ["stream_to_gradio", "GradioUI"]
