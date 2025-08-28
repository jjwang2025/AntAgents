#!/usr/bin/env python
# coding=utf-8

import os
import re
import shutil
from pathlib import Path
from typing import Generator

from antagents.agent_types import AgentAudio, AgentImage, AgentText
from antagents.agents import MultiStepAgent, PlanningStep
from antagents.memory import ActionStep, FinalAnswerStep
from antagents.models import ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from antagents.utils import _is_package_available, decode_unicode_escapes


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    """获取步骤日志的脚注字符串，包含持续时间和令牌信息"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


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
    return model_output.strip()


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
    step_number = f"Step {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=f"**{step_number}**", metadata={"status": "done"})

    # 首先生成LLM的思考/推理
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=model_output, metadata={"status": "done"})

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
            content = _format_code_content(content)

        # 创建工具调用消息
        parent_message_tool = gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "title": f"🛠️ Used tool {first_tool_call.name}",
                "status": "done",
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
                content=f"```bash\n{log_content}\n",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

    # 显示观察中的任何图像
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                metadata={"title": "🖼️ Output Image", "status": "done"},
            )

    # 处理错误
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=str(step_log.error), metadata={"title": "💥 Error", "status": "done"}
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
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="**Planning step**", metadata={"status": "done"})
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=step_log.plan, metadata={"status": "done"})
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, "Planning step"),
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
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"status": "done"},
        )
    else:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=f"**Final answer:** {str(final_answer)}", metadata={"status": "done"}
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
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
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
        from antagents import CodeAgent, GradioUI, InferenceClientModel

        model = InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        agent = CodeAgent(tools=[], model=model)
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
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)

    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        # 从模板智能体获取智能体类型
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt, metadata={"status": "done"}))
            yield messages

            for msg in stream_to_gradio(
                session_state["agent"], task=prompt, reset_agent_memory=self.reset_agent_memory
            ):
                if isinstance(msg, gr.ChatMessage):
                    messages[-1].metadata["status"] = "done"
                    messages.append(msg)
                elif isinstance(msg, str):  # 那么它只是一个完成增量
                    msg = msg.replace("<", r"\<").replace(">", r"\>")  # HTML标签似乎会破坏Gradio Chatbot
                    if messages[-1].metadata["status"] == "pending":
                        messages[-1].content = msg
                    else:
                        messages.append(
                            gr.ChatMessage(role=MessageRole.ASSISTANT, content=msg, metadata={"status": "pending"})
                        )
                yield messages

            yield messages
        except Exception as e:
            yield messages
            raise gr.Error(f"Error in interaction: {str(e)}")

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

    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr

        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

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

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # 添加会话状态以存储会话特定数据
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> 此Web界面允许您与一个可以使用工具并执行步骤以完成任务的小型智能体进行交互。"
                    + (f"\n\n**智能体描述:**\n{self.description}" if self.description else "")
                )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                # 如果提供了上传文件夹，则启用上传功能
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                gr.HTML(
                    "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/antagents'><b>antagents</b></a></center></h4>"
                )

            # 主聊天界面
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/antagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
                latex_delimiters=[
                    {"left": r"$$", "right": r"$$", "display": True},
                    {"left": r"$", "right": r"$", "display": False},
                    {"left": r"\[", "right": r"\]", "display": True},
                    {"left": r"\(", "right": r"\)", "display": False},
                ],
            )

            # 设置事件处理器
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            chatbot.clear(self.agent.memory.reset)
        return demo


__all__ = ["stream_to_gradio", "GradioUI"]