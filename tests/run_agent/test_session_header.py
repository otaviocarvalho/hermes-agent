"""WS-2.3: AIAgent._create_openai_client must inject X-Session-Id.

C5 (spec .specs/features/2.3-session-header): the choke point merges the
agent's opaque session header into default_headers on the per-call kwargs
copy — never mutating the stored kwargs dict (#10933 contract) — and an
explicit X-Session-Id passed by the caller always wins.
"""

from unittest.mock import patch

from run_agent import AIAgent

from agent.session_identity import SESSION_HEADER_NAME


def _make_agent(**extra):
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **extra,
    )


class _FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        _FakeOpenAI.instances.append(self)

    def close(self):
        pass


def test_inject_preserve():
    """Injection lands on the constructed client, stored kwargs stay
    unmutated, and an explicit header wins over the derived one."""
    _FakeOpenAI.instances = []
    agent = _make_agent()

    # sanity: agent derived an opaque header from its CLI session id
    assert agent._session_header
    assert agent._session_header.startswith("s-")

    stored_kwargs = {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    with patch("run_agent.OpenAI", _FakeOpenAI):
        agent._create_openai_client(stored_kwargs, reason="test", shared=False)

    constructed = _FakeOpenAI.instances[-1]._kwargs
    assert constructed["default_headers"][SESSION_HEADER_NAME] == agent._session_header
    # #10933 contract: the caller's stored dict was not mutated
    assert "default_headers" not in stored_kwargs

    # explicit header wins — derived value must not overwrite it
    explicit_kwargs = {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "default_headers": {SESSION_HEADER_NAME: "caller-explicit"},
    }
    with patch("run_agent.OpenAI", _FakeOpenAI):
        agent._create_openai_client(explicit_kwargs, reason="test", shared=False)

    constructed = _FakeOpenAI.instances[-1]._kwargs
    assert constructed["default_headers"][SESSION_HEADER_NAME] == "caller-explicit"
    # merge copied — the caller's dict still holds the explicit value untouched
    assert explicit_kwargs["default_headers"][SESSION_HEADER_NAME] == "caller-explicit"


def test_gateway_key_beats_session_id_for_identity():
    """gateway_session_key (stable topic key) is the identity source; two
    agents with the same topic key share a header even when their CLI
    session ids differ, and a different topic gets a different header."""
    agent_a = _make_agent(gateway_session_key="agent:main:telegram:dm:123", session_id="sessAAA")
    agent_b = _make_agent(gateway_session_key="agent:main:telegram:dm:123", session_id="sessBBB")
    agent_c = _make_agent(gateway_session_key="agent:main:telegram:dm:999", session_id="sessAAA")
    assert agent_a._session_header == agent_b._session_header
    assert agent_a._session_header != agent_c._session_header
