"""WS-2.3 session identity tests (ADR-0006).

Covers the identity helper semantics the spec pins:
- AC2/C1 same_topic: interleaved topics keep each topic's own stable value
- AC3/C2 different_topics: distinct topics never collide
- AC4/C3 graceful: unset/invalid identity degrades to no header, never raises
- C4 deterministic_opaque: value is restart-stable, per-key distinct, opaque
"""

import hashlib

import pytest

from agent.session_identity import (
    SESSION_HEADER_NAME,
    clear_session_header_value,
    get_session_header_value,
    merge_session_header,
    session_header_value,
    set_session_header_value,
)


@pytest.fixture(autouse=True)
def _clean_context():
    clear_session_header_value()
    yield
    clear_session_header_value()


# ---------------------------------------------------------------------------
# C1 - AC2 same_topic
# ---------------------------------------------------------------------------

def test_same_topic_interleaved_keeps_value():
    """Two topics' calls interleaved: each read/injection still resolves to
    its own topic's value — the ContextVar handoff does not bleed."""
    value_a = session_header_value("agent:main:telegram:dm:123")
    value_b = session_header_value("agent:main:telegram:group:456")
    assert value_a and value_b

    set_session_header_value(value_a)
    assert get_session_header_value() == value_a

    # topic B's turn interleaves
    set_session_header_value(value_b)
    headers_b = merge_session_header({}, get_session_header_value())
    assert headers_b[SESSION_HEADER_NAME] == value_b

    # topic A's turn comes back — same value as before the interleave
    set_session_header_value(value_a)
    headers_a = merge_session_header({}, get_session_header_value())
    assert headers_a[SESSION_HEADER_NAME] == value_a
    assert headers_a[SESSION_HEADER_NAME] != headers_b[SESSION_HEADER_NAME]


def test_same_topic_merge_repeated_is_stable():
    """Repeated merges for one topic produce the identical header each time
    (idempotent per conversation)."""
    value = session_header_value("topic-x")
    first = merge_session_header({}, value)
    second = merge_session_header({"Authorization": "Bearer t"}, value)
    assert first[SESSION_HEADER_NAME] == second[SESSION_HEADER_NAME] == value
    assert second["Authorization"] == "Bearer t"


# ---------------------------------------------------------------------------
# C2 - AC3 different_topics
# ---------------------------------------------------------------------------

def test_different_topics_get_different_values():
    keys = [
        "agent:main:telegram:dm:123",
        "agent:main:telegram:dm:124",
        "agent:main:telegram:group:456",
        "cli-session-abc",
    ]
    values = [session_header_value(k) for k in keys]
    assert all(values)
    assert len(set(values)) == len(values)


def test_topic_value_differs_when_topic_key_differs_suffix():
    assert session_header_value("agent:main:telegram:dm:1:t1") != session_header_value(
        "agent:main:telegram:dm:1:t2"
    )


# ---------------------------------------------------------------------------
# C3 - AC4 graceful
# ---------------------------------------------------------------------------

def test_graceful_no_identity_no_header():
    """No identity key → helper returns None and merge produces no header;
    nothing raises, the call would simply proceed headerless."""
    assert session_header_value(None) is None
    assert session_header_value("") is None
    assert session_header_value(12345) is None  # non-string identity

    unset_context = get_session_header_value()
    assert unset_context is None

    merged = merge_session_header({"X-Trace": "t"}, None)
    assert SESSION_HEADER_NAME not in merged
    assert merged == {"X-Trace": "t"}


def test_graceful_merge_with_bad_inputs_never_raises():
    """Identity helper cannot throw into the call path: hostile shapes
    degrade instead of raising. ``None`` headers is the normal absent case
    → fresh dict carrying just the header."""
    assert merge_session_header(None, "s-abc") == {SESSION_HEADER_NAME: "s-abc"}
    assert merge_session_header("not-a-dict", "s-abc") == {}
    assert merge_session_header({}, None) == {}


def test_graceful_unset_context_clear():
    set_session_header_value("s-abc")
    assert get_session_header_value() == "s-abc"
    clear_session_header_value()
    assert get_session_header_value() is None


# ---------------------------------------------------------------------------
# C4 - deterministic + opaque
# ---------------------------------------------------------------------------

def test_deterministic_across_repeated_calls():
    """Same key → same value on every call (pure function of the key: the
    'restart' property — no process state participates)."""
    key = "agent:main:telegram:dm:123"
    values = {session_header_value(key) for _ in range(5)}
    assert len(values) == 1


def test_opaque_value_shape_and_no_key_leak():
    key = "agent:main:telegram:dm:12345-secret-suffix"
    value = session_header_value(key)
    assert value.startswith("s-")
    assert len(value) == len("s-") + 16
    assert value[2:] == hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    # raw key bytes never appear in the wire value
    assert key not in value
    assert "12345" not in value
    assert "secret" not in value
