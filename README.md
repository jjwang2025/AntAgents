# AntAgents
Lightweight Framework for Agent Swarms

## How to install
Run the below command line tools to install.
```
git clone https://github.com/jjwang2025/AntAgents.git
cd AntAgents
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Compatibility Notes

If you are using GPT-5 or other reasoning models, see `docs/REASONING_MODEL_COMPAT.md` for the new dual-stack OpenAI compatibility changes and usage notes.

Quick verification scripts:

- `python recipes/react_tool_use.py`: textbook multi-step ReAct example with minimal local tools
- `python recipes/plan_and_execute.py`: textbook planner-executor-synthesizer example
- `python recipes/test_responses_streaming.py`: local responses streaming compatibility check

Set `OPENAI_API_MODE=auto` to let GPT-5 / reasoning models use the `responses` API automatically.

ENJOY IY! 😃
