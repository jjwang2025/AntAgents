#!/usr/bin/env python
# coding=utf-8

import importlib
import json
import os
import re
import tempfile
import textwrap
import time
import warnings
from datetime import datetime
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

import yaml
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .memory import (
    ActionStep,
    AgentMemory,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from .models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    MessageRole,
    Model,
    agglomerate_stream_deltas,
    parse_json_if_needed,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .tools import BaseTool, Tool, validate_tool_arguments
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    create_agent_gradio_app_template,
    is_valid_name,
    make_init_file,
    truncate_content,
    decode_unicode_escapes,
)


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: dict[str, Any]) -> str:
    # 在现有variables中添加datetime信息
    now = datetime.now()
    variables_with_datetime = variables.copy()  # 复制原有variables
    variables_with_datetime['current_datetime'] = now.strftime("%B %d, %Y at %I:%M:%S %p")  # 添加datetime
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables_with_datetime)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class ActionOutput:
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    """
    规划步骤的提示模板。

    Args:
        plan (`str`): 初始规划提示。
        update_plan_pre_messages (`str`): 更新规划前的消息提示。
        update_plan_post_messages (`str`): 更新规划后的消息提示。
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    托管智能体的提示模板。

    Args:
        task (`str`): 任务提示。
        report (`str`): 报告提示。
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    最终答案的提示模板。

    Args:
        pre_messages (`str`): 消息前的提示。
        post_messages (`str`): 消息后的提示。
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    智能体的提示模板。

    Args:
        system_prompt (`str`): 系统提示。
        planning ([`~agents.PlanningPromptTemplate`]): 规划提示模板。
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): 托管智能体提示模板。
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): 最终答案提示模板。
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    """保存智能体运行的扩展信息。

    Attributes:
        output (Any | None): 智能体运行的最终输出，如果可用。
        state (Literal["success", "error"]): 运行后智能体的最终状态。
        messages (list[dict]): 智能体的记忆，作为消息列表。
        token_usage (TokenUsage | None): 运行期间使用的令牌计数。
        timing (Timing): 智能体运行的计时详情：开始时间、结束时间、持续时间。
    """

    output: Any | None
    state: Literal["success", "error"]
    messages: list[dict]
    token_usage: TokenUsage | None
    timing: Timing


StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]


class MultiStepAgent(ABC):
    """
    智能体类，使用ReAct框架逐步解决给定任务：
    在未达到目标时，智能体将执行一个由LLM给出的动作循环，并从环境中获取观察结果。

    Args:
        tools (`list[Tool]`): 智能体可以使用的工具列表。
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): 生成智能体动作的模型。
        prompt_templates ([`~agents.PromptTemplates`], *optional*): 提示模板。
        instructions (`str`, *optional*): 智能体的自定义指令，将插入系统提示中。
        max_steps (`int`, default `20`): 智能体解决任务的最大步数。
        add_base_tools (`bool`, default `False`): 是否将基础工具添加到智能体的工具中。
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): 智能体日志的详细级别。
        grammar (`dict[str, str]`, *optional*): 用于解析LLM输出的语法。
            <Deprecated version="1.17.0">
            参数`grammar`已弃用，将在1.20版本中移除。
            </Deprecated>
        managed_agents (`list`, *optional*): 智能体可以调用的托管智能体。
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable | list[Callable]]`, *optional*): 每步执行的回调函数。
        planning_interval (`int`, *optional*): 智能体运行规划步骤的间隔。
        name (`str`, *optional*): 仅对托管智能体必要 - 调用此智能体的名称。
        description (`str`, *optional*): 仅对托管智能体必要 - 此智能体的描述。
        provide_run_summary (`bool`, *optional*): 作为托管智能体调用时是否提供运行摘要。
        final_answer_checks (`list[Callable]`, *optional*): 在接受最终答案前运行的验证函数列表。
            每个函数应：
            - 以最终答案和智能体的记忆为参数。
            - 返回布尔值表示最终答案是否有效。
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        if grammar is not None:
            warnings.warn(
                "Parameter 'grammar' is deprecated and will be removed in version 1.20.",
                FutureWarning,
            )
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self._setup_step_callbacks(step_callbacks)
        self.stream_outputs = False

    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """设置托管智能体并配置日志"""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # 确保托管智能体可以作为工具被模型调用：设置其输入和输出类型
            for agent in self.managed_agents.values():
                agent.inputs = {
                    "task": {"type": "string", "description": "Long detailed description of the task."},
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                    },
                }
                agent.output_type = "string"

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, BaseTool) for tool in tools), (
            "All elements must be instance of BaseTool (or a subclass)"
        )
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        # 初始化步骤回调注册表
        self.step_callbacks = CallbackRegistry()
        if step_callbacks:
            # 为向后兼容，仅为ActionStep注册回调列表
            if isinstance(step_callbacks, list):
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            # 为特定步骤类注册回调字典
            elif isinstance(step_callbacks, dict):
                for step_cls, callbacks in step_callbacks.items():
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                raise ValueError("step_callbacks must be a list or dict")
        # 为向后兼容，仅为ActionStep注册monitor的update_metrics
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        运行智能体执行给定任务。

        Args:
            task (`str`): 要执行的任务。
            stream (`bool`): 是否以流模式运行。
                如果为`True`，返回一个生成器，逐步生成每个执行步骤。必须迭代此生成器以处理各个步骤（例如使用for循环或`next()`）。
                如果为`False`，内部执行所有步骤并仅在完成后返回最终答案。
            reset (`bool`): 是否重置对话或从之前的运行继续。
            images (`list[PIL.Image.Image]`, *optional*): 图像对象列表。
            additional_args (`dict`, *optional*): 任何其他要传递给智能体运行的变量，例如图像或数据框。请为它们提供清晰的名称！
            max_steps (`int`, *optional*): 智能体解决任务的最大步数。如果未提供，将使用智能体的默认值。
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # 步骤通过生成器逐步返回以便迭代
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        run_start_time = time.time()
        # 仅在最后返回输出。我们只看最后一步

        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        if self.return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            # 即使达到最大步数，只要成功生成了最终答案，就认为是成功状态
            if (self.memory.steps and 
                hasattr(self.memory.steps[-1], "error") and 
                self.memory.steps[-1].error is not None and
                not isinstance(self.memory.steps[-1].error, AgentMaxStepsError)):
                state = "error"
            else:
                state = "success"  # 包括达到最大步数但成功生成最终答案的情况

            messages = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                messages=messages,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
            self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
        ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """执行流式运行的智能体步骤
        
        Args:
            task: 要执行的任务
            max_steps: 最大步数限制
            images: 可选的图像输入列表
            
        Yields:
            生成器返回各步骤的执行结果
        """
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # 如果设置了规划间隔，则在第一步或间隔步数时执行规划步骤
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    task, is_first_step=len(self.memory.steps) == 1, step=self.step_number
                ):  # 这里不使用step_number属性，因为可能有之前运行的步骤
                    yield element
                    planning_step = element
                assert isinstance(planning_step, PlanningStep)  # 最后生成的元素应该是PlanningStep
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            # 开始执行动作步骤
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in self._step_stream(action_step):
                    # 生成所有输出
                    yield output
                    
                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
                        returned_final_answer = True
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # 智能体生成错误不是由模型错误引起的，而是实现错误：应该抛出并退出
                raise e
            except AgentError as e:
                # 其他智能体错误类型是由模型引起的，应该记录并继续
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task, images)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    def _validate_final_answer(self, final_answer: Any):
        """验证最终答案是否符合检查条件"""
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep | PlanningStep):
        """完成步骤的最终处理"""
        memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        """处理达到最大步数的情况 - 生成最终答案而不是抛出异常"""
        action_step_start_time = time.time()
        
        # 生成最终答案
        final_answer_message = self.provide_final_answer(task, images)
        final_answer = final_answer_message.content
        
        # 创建正常的内存步骤
        final_memory_step = ActionStep(
            step_number=self.step_number,
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer_message.token_usage,
        )
        final_memory_step.action_output = final_answer
        
        # 记录信息但不作为错误
        self.logger.log(
            Text(f"Reached max steps ({self.max_steps}), generating final answer based on current knowledge.", 
                 style=f"bold {YELLOW_HEX}"),
            level=LogLevel.INFO,
        )
        
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        """生成规划步骤"""
        start_time = time.time()
        if is_first_step:
            # 初始规划消息构造
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            # 解码 Unicode 转义字符
                            decoded_text = decode_unicode_escapes(plan_message_content)
                            # 更新命令行的显示内容
                            live.update(Markdown(decoded_text))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
                    if plan_message.token_usage
                    else (None, None)
                )
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # 更新规划模式，移除系统提示和之前的规划消息
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            )
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            # 解码 Unicode 转义字符
                            decoded_text = decode_unicode_escapes(plan_message_content)
                            # 更新命令行的显示内容
                            live.update(Markdown(decoded_text))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                if plan_message.token_usage is not None:
                    input_tokens, output_tokens = (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @property
    def logs(self):
        """已弃用的日志属性"""
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """子类需要实现的抽象方法"""
        ...

    def interrupt(self):
        """中断智能体执行"""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        将记忆中的历史记录转换为消息列表
        
        Args:
            summary_mode: 是否使用摘要模式
            
        Returns:
            可用于LLM输入的消息列表
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        执行ReAct框架中的一个步骤：智能体思考、行动并观察结果
        
        Yields:
            生成器返回流式输出
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        """
        执行ReAct框架中的一个步骤
        
        Returns:
            返回步骤结果或最终答案
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        从LLM输出中解析动作
        
        Args:
            model_output: LLM输出文本
            split_token: 动作分隔符
            
        Returns:
            返回(推理, 动作)元组
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # 从末尾开始索引可以处理输出中包含多个分隔符的情况
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> ChatMessage:
        """
        根据智能体交互记录提供最终答案
        
        Args:
            task: 要执行的任务
            images: 可选的图像输入
            
        Returns:
            返回包含最终答案的聊天消息
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        if images:
            messages[0].content += [{"type": "image", "image": image} for image in images]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )

    def visualize(self):
        """创建智能体结构的可视化树"""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """打印智能体步骤的回放
        
        Args:
            detailed: 是否显示每一步的详细内存
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """托管智能体的调用方法"""
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer


class ToolCallingAgent(MultiStepAgent):
    """
    使用JSON式工具调用的智能体
    
    Args:
        tools: 智能体可用的工具列表
        model: 生成智能体动作的模型
        prompt_templates: 提示模板
        planning_interval: 规划步骤的执行间隔
        stream_outputs: 是否流式输出
        max_tool_threads: 并行工具调用的最大线程数
        **kwargs: 其他关键字参数
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("antagents.prompts").joinpath("toolcalling_agent.yaml").read_text(encoding='utf-8')
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        # 流式输出设置
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # 工具调用设置
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        """返回工具和托管智能体的合并列表"""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        """初始化系统提示"""
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """执行工具调用智能体的步骤流"""
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # 在日志中添加新步骤
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        # 获取聚合后的消息并转换为Markdown文本
                        markdown_text = agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown()
                        # 解码 Unicode 转义字符
                        decoded_text = decode_unicode_escapes(markdown_text)
                        # 更新命令行的显示内容
                        live.update(Markdown(decoded_text))
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                if chat_message.content is None and chat_message.raw is not None:
                    log_content = str(chat_message.raw)
                else:
                    log_content = str(chat_message.content) or ""

                self.logger.log_markdown(
                    content=log_content,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # 记录模型输出
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # 管理状态变量
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """处理模型输出的工具调用并更新智能体记忆"""
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # 处理单个工具调用的辅助函数
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # 转义可能的富文本标签
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # 并行处理工具调用
        outputs = {}
        if len(parallel_calls) == 1:
            # 如果只有一个调用，直接处理
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # 多个工具调用并行处理
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [
                    executor.submit(process_single_tool_call, tool_call) for tool_call in parallel_calls.values()
                ]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """用状态变量替换参数中的字符串值"""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        执行工具或托管智能体
        
        Args:
            tool_name: 工具或托管智能体名称
            arguments: 调用参数
        """
        # 检查工具是否存在
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # 获取工具并替换参数中的状态变量
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

        try:
            # 使用适当参数调用工具
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            else:
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)

        except Exception as e:
            # 处理执行错误
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e
