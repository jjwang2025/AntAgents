#!/usr/bin/env python3

import sys
import importlib.util
import types
from pathlib import Path
from types import SimpleNamespace

try:
    import rich  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: rich. Install requirements.txt first, then run this recipe again."
    ) from exc

# Avoid importing antagents.__init__, which eagerly imports optional modules unrelated to this check.
package_root = Path(__file__).resolve().parents[1] / "src" / "antagents"
package = types.ModuleType("antagents")
package.__path__ = [str(package_root)]
sys.modules.setdefault("antagents", package)

models_path = package_root / "models.py"
spec = importlib.util.spec_from_file_location("antagents.models", models_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["antagents.models"] = module
spec.loader.exec_module(module)

OpenAIServerModel = module.OpenAIServerModel
agglomerate_stream_deltas = module.agglomerate_stream_deltas


def main() -> None:
    model = OpenAIServerModel(
        model_id="gpt-5",
        api_base="https://example.com",
        api_key="test",
        api_mode="responses",
    )

    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Planning...", item_id="r1", output_index=0),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(type="function_call", call_id="call_1", id="item_1", name="final_answer"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", output_index=1, delta='{"answer":"he'),
        SimpleNamespace(type="response.function_call_arguments.delta", output_index=1, delta='llo"}'),
        SimpleNamespace(type="response.output_text.delta", delta="done"),
        SimpleNamespace(type="response.web_search_call.in_progress", item_id="w1", output_index=2),
        SimpleNamespace(type="response.web_search_call.completed", item_id="w1", output_index=2),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=5)),
        ),
    ]

    deltas = []
    for event in events:
        delta = model._responses_stream_event_to_delta(event)
        if delta is not None:
            deltas.append(delta)

    builtin_events = [
        event
        for delta in deltas
        for event in (delta.builtin_tool_events or [])
    ]

    message = agglomerate_stream_deltas(deltas)

    assert "Planning..." in (message.content or "")
    assert "done" in (message.content or "")
    assert builtin_events
    assert builtin_events[0].tool_type == "web_search_call"
    assert builtin_events[0].status == "in_progress"
    assert builtin_events[0].item_id == "w1"
    assert builtin_events[1].status == "completed"
    assert "[web_search_call] in_progress (w1)" in (message.content or "")
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].function.name == "final_answer"
    assert message.tool_calls[0].function.arguments == '{"answer":"hello"}'
    assert message.token_usage is not None
    assert message.token_usage.input_tokens == 3
    assert message.token_usage.output_tokens == 5

    print("responses streaming compatibility test passed")


if __name__ == "__main__":
    main()
