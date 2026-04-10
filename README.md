# AntAgents

轻量级多智能体框架，支持工具调用、规划执行、Gradio 交互界面，以及 GPT-5 / reasoning models 的 `responses` 协议兼容。

## 安装

请按下面顺序安装：

```bash
git clone https://github.com/jjwang2025/AntAgents.git
cd AntAgents
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 环境变量

仓库当前示例主要沿用以下环境变量命名：

- `DEEPSEEK_MODEL_ID`
- `DEEPSEEK_URL`
- `DEEPSEEK_API_KEY`

如果你使用 GPT-5 或其他 reasoning models，建议额外设置：

```bash
OPENAI_API_MODE=auto
```

这样框架会自动为 reasoning models 切换到 `responses` 协议。

## 快速开始

### 1. 教科书式 ReAct 示例

```bash
python recipes/react_tool_use.py
```

特点：
- 使用最小本地工具
- 展示多步 `Action -> Observation -> Final Answer`
- 适合理解 ReAct 闭环

### 2. 教科书式 Plan-and-Execute 示例

```bash
python recipes/plan_and_execute.py
```

特点：
- 分为 Planner / Executor / Synthesizer 三阶段
- 适合理解规划执行范式

### 3. `responses` 流式兼容测试

```bash
python recipes/test_responses_streaming.py
```

特点：
- 不调用真实 API
- 验证 `responses` 流式事件到内部结构的归一化

### 4. 模型调用综合示例

```bash
python recipes/run_model.py
```

特点：
- 演示非流式与流式调用
- 已兼容 Windows 非交互终端
- 可展示 `responses` built-in tool 的结构化事件

### 5. Gradio UI 示例

```bash
python recipes/gradio_ui.py
```

特点：
- 图形化展示智能体步骤轨迹
- 当前支持规划步骤、工具调用、执行日志、最终答案
- 对 `responses` built-in tool 事件做了更清晰的展示

## GPT-5 / Reasoning Models

如果你正在使用 GPT-5、`o1`、`o3`、`o4-mini` 等 reasoning models，请优先阅读：

- `docs/REASONING_MODEL_COMPAT.md`

该文档说明了：
- `chat_completions` 与 `responses` 的双栈兼容
- 流式 tool calling 的适配情况
- built-in tool 事件展示支持范围

## 示例分类

当前推荐从这几个示例开始：

- `recipes/react_tool_use.py`：ReAct
- `recipes/plan_and_execute.py`：Plan-and-Execute
- `recipes/run_model.py`：模型基础调用
- `recipes/gradio_ui.py`：Web UI 交互
- `recipes/test_responses_streaming.py`：协议兼容测试

## 文档

- `AGENTS.md`：给 OpenCode / 代码智能体的仓库注意事项
- `docs/REASONING_MODEL_COMPAT.md`：GPT-5 / reasoning models 兼容说明
- `docs/ROADMAP.md`：后续功能提升计划

## 当前特性

- 兼容 OpenAI-compatible API
- 支持 `chat.completions` 与 `responses` 双协议
- 对 reasoning models 自动切换精简 prompt 模板
- 支持多步 ReAct 示例
- 支持三阶段 Plan-and-Execute 示例
- 支持 `responses` built-in tool 事件结构化归一化
- 提供 Gradio 界面用于可视化智能体轨迹
