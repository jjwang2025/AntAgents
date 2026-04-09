# GPT-5 / Reasoning Model Compatibility

## 背景

本仓库原先默认基于旧版 `chat.completions` + `tool_calls` 协议实现工具调用。

从 GPT-5 及新一代 reasoning models 开始，OpenAI 的推荐调用方式已经转向 `responses` API，工具调用和响应结构与旧协议存在明显差异。直接沿用旧实现时，常见问题包括：

- reasoning 模型无法正确返回工具调用
- `stop` 等旧参数与新模型不兼容
- agent 在没有原生 `tool_calls` 时误把普通文本答案当成解析失败

本次修改的目标是：

- 保留旧模型的 `chat.completions` 兼容性
- 为 GPT-5 / o 系 reasoning models 增加 `responses` 路径
- 在 `ToolCallingAgent` 层尽量不改内部抽象，统一收敛到现有 `ChatMessage` / `tool_calls` 结构

## 本次修改概览

### 1. `src/antagents/models.py`

对 `OpenAIServerModel` 做了双栈协议改造。

新增能力：

- `api_mode="auto" | "chat_completions" | "responses"`
- `auto` 模式下根据 `model_id` 自动切换协议

当前自动切换规则：

- `gpt-5*` / `o1*` / `o3*` / `o4-mini*` -> `responses`
- 其他模型 -> `chat.completions`

新增的主要内部方法：

- `is_reasoning_model()`
- `_uses_responses_api()`
- `_prepare_chat_completions_kwargs()`
- `_prepare_responses_kwargs()`
- `_normalize_chat_completions_response()`
- `_normalize_responses_response()`
- `get_responses_tool_json_schema()`

### 2. `src/antagents/agents.py`

调整了 `ToolCallingAgent._step_stream()` 的兜底逻辑。

旧行为：

- 只要没有原生 `tool_calls`，就强行把文本交给 `parse_tool_calls()`

新行为：

- 如果没有原生 `tool_calls`
- 且文本不是旧式 JSON 工具调用
- 则把该文本视为普通 assistant 最终答案返回

这样 reasoning 模型即使直接输出答案，也不会被误判成“工具调用解析失败”。

### 3. `src/antagents/cli.py`

CLI 新增参数：

```bash
--api-mode auto|chat_completions|responses
```

用途：

- `auto`: 默认推荐，按模型自动切换
- `chat_completions`: 强制旧协议
- `responses`: 强制新协议

## 兼容后的行为

### 旧模型

例如：

- `deepseek-chat`
- 大多数 OpenAI-compatible 老模型

行为：

- 继续走 `chat.completions`
- 继续使用原有 `tool_calls`
- 对现有 agent 逻辑基本无影响

### GPT-5 / reasoning models

例如：

- `gpt-5`
- `o1`
- `o3`
- `o4-mini`

行为：

- 默认走 `responses`
- `responses.output` 中的 `message` / `function_call` 会被归一化为当前框架内部使用的：
- `ChatMessage.content`
- `ChatMessage.tool_calls`
- `TokenUsage`

## 当前限制

### 1. `responses` 流式尚未实现

目前 reasoning 模型只保证非流式路径可用。

如果调用了 `generate_stream()` 且当前模型被判定为 `responses` 路径，会抛出明确错误：

```text
Streaming for responses API is not implemented yet. Use stream=False for reasoning models.
```

### 2. 本轮优先支持 function tool call

当前 `responses` 归一化逻辑重点覆盖：

- 普通文本输出
- `function_call` 输出

这已经满足当前 `ToolCallingAgent` 的主流程。

对更复杂的 built-in tool event / 流式 event 还没有做完整适配。

### 3. `stop` 参数已按 reasoning 模型禁用

本次将 `supports_stop_parameter()` 调整为：

- reasoning 模型统一视为不支持 `stop`

避免 GPT-5 路径继续带入旧参数。

## 使用示例

### 1. 默认自动切换

```bash
python src/antagents/cli.py "你好" --model-id gpt-5 --api-base <API_BASE> --api-key <API_KEY> --api-mode auto
```

### 2. 强制旧协议

```bash
python src/antagents/cli.py "你好" --model-id deepseek-chat --api-mode chat_completions
```

### 3. 强制新协议

```bash
python src/antagents/cli.py "测试工具调用" --model-id gpt-5 --api-base <API_BASE> --api-key <API_KEY> --api-mode responses
```

## 验证情况

本次修改后已完成以下验证：

- 修改文件通过 `py_compile`
- `gpt-5` 会自动切到 `responses`
- `deepseek-chat` 会继续走 `chat.completions`
- `responses` 路径的工具 schema 已按当前 SDK 结构转换
- 本地 smoke test 已验证 `responses` 响应能归一化为框架内部的 `ChatMessage.tool_calls`

## 后续建议

如果继续完善，建议优先做这两项：

1. 为 `responses` 实现流式事件到 `ChatMessageStreamDelta` 的映射
2. 给 `recipes/helloworld.py`、`recipes/run_model.py` 增加 `api_mode` 示例，方便直接验证 reasoning 模型
