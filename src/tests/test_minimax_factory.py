import json

import httpx
import pytest
from agno.models.message import Message

from src.utils.llm.factory import get_model


@pytest.mark.parametrize(
    ("model_id", "mode", "expected_thinking"),
    [
        ("MiniMax-M3", None, {"type": "adaptive"}),
        ("MiniMax-M3", "adaptive", {"type": "adaptive"}),
        ("MiniMax-M3", "disabled", {"type": "disabled"}),
        ("MiniMax-M3", "invalid", {"type": "adaptive"}),
        ("MiniMax-M2.7", None, None),
        ("MiniMax-M2.7", "always_on", None),
        ("MiniMax-M2.7", "disabled", None),
    ],
)
def test_minimax_thinking_request(monkeypatch, model_id, mode, expected_thinking):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.setenv("MINIMAX_REGION", "global_en")
    if mode is None:
        monkeypatch.delenv("MINIMAX_THINKING", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_THINKING", mode)

    requests = []

    def handle_request(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        model = get_model("minimax", model_id, http_client=client)
        model.invoke(
            [Message(role="user", content="Hello")], Message(role="assistant")
        )

    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == model_id
    assert "enable_thinking" not in body
    if expected_thinking is None:
        assert "thinking" not in body
    else:
        assert body["thinking"] == expected_thinking
