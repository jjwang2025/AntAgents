import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from antagents.models import ChatMessage, MessageRole
from antagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from antagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from antagents.models import ChatMessage
    from antagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]  # 导出的公共接口


logger = getLogger(__name__)


@dataclass
class ToolCall:
    """工具调用类"""
    name: str
    arguments: Any
    id: str

    def dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    """记忆步骤基类"""
    def dict(self):
        """转换为字典格式"""
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """转换为消息列表"""
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """动作步骤"""
    step_number: int  # 步骤编号
    timing: Timing  # 计时信息
    model_input_messages: list[ChatMessage] | None = None  # 模型输入消息
    tool_calls: list[ToolCall] | None = None  # 工具调用列表
    error: AgentError | None = None  # 错误信息
    model_output_message: ChatMessage | None = None  # 模型输出消息
    model_output: str | list[dict[str, Any]] | None = None  # 模型输出
    observations: str | None = None  # 观察结果
    observations_images: list["PIL.Image.Image"] | None = None  # 观察图片
    action_output: Any = None  # 动作输出
    token_usage: TokenUsage | None = None  # token使用情况
    is_final_answer: bool = False  # 是否为最终答案

    def dict(self):
        """转换为字典格式"""
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """转换为消息列表"""
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    """规划步骤"""
    model_input_messages: list[ChatMessage]  # 模型输入消息
    model_output_message: ChatMessage  # 模型输出消息
    plan: str  # 计划内容
    timing: Timing  # 计时信息
    token_usage: TokenUsage | None = None  # token使用情况

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """转换为消息列表"""
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # 用于防止模型简单延续计划消息
        ]


@dataclass
class TaskStep(MemoryStep):
    """任务步骤"""
    task: str  # 任务内容
    task_images: list["PIL.Image.Image"] | None = None  # 任务图片

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """转换为消息列表"""
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    """系统提示步骤"""
    system_prompt: str  # 系统提示内容

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """转换为消息列表"""
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    """最终答案步骤"""
    output: Any  # 输出结果


class AgentMemory:
    """智能体记忆类，用于存储系统提示和所有执行步骤"""

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """重置记忆，保留系统提示"""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """获取简洁步骤信息（不含模型输入消息）"""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """获取完整步骤信息（含模型输入消息）"""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """重放智能体执行步骤"""
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
                    if detailed and step.model_output_message is not None:
                        logger.log_messages([step.model_output_message], level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)


class CallbackRegistry:
    """回调注册表，用于在智能体执行步骤时调用回调函数"""

    def __init__(self):
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}  # 回调函数存储

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """注册回调函数"""
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """执行回调函数"""
        # 兼容旧版回调
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)