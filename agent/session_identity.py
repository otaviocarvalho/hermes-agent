"""WS-2.3 stable session identity for provider calls (ADR-0006).

Every OpenAI-compatible provider call carries ``X-Session-Id``: an opaque,
deterministic token that is stable for one conversation topic (gateway
``session_key``) or one CLI session (``session_id``). Providers can use it
for session affinity / cold-start routing; its absence must never break a
call — routing just degrades to a cold session.

Design notes:
- Identity source is ``gateway_session_key`` (stable per topic, survives
  context resets) falling back to ``session_id``. We deliberately do NOT
  hash transcripts: compaction and prompt edits would rotate the value
  mid-session and break affinity exactly when it matters (ADR-0006).
- The wire value is ``s-`` + first 16 hex chars of SHA-256: deterministic
  across restarts, different per key, and reveals nothing about the raw
  key (which may contain chat IDs).
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
from typing import Dict, Optional

SESSION_HEADER_NAME = "X-Session-Id"
_SESSION_HEADER_NAME_LOWER = SESSION_HEADER_NAME.lower()

# Request-scoped value; set by AIAgent.__init__ (and optionally gateway
# helpers) so constructions that bypass the agent instance — e.g. the
# module-level OpenAI proxy in agent/auxiliary_client.py — can still tag
# their clients with the current conversation's header.
_session_header_ctx: ContextVar[Optional[str]] = ContextVar(
    "ws23_session_header", default=None
)


def session_header_value(identity_key: Optional[str]) -> Optional[str]:
    """Derive the opaque wire value for a session identity key.

    Deterministic: same key → same value across restarts. Different keys
    → different values. Empty/None key → None (caller sends no header).
    """
    if not identity_key or not isinstance(identity_key, str):
        return None
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:16]
    return f"s-{digest}"


def set_session_header_value(value: Optional[str]) -> None:
    """Publish the current conversation's header value request-wide."""
    _session_header_ctx.set(value)


def get_session_header_value() -> Optional[str]:
    """Current conversation's header value, or None if unset."""
    return _session_header_ctx.get()


def clear_session_header_value() -> None:
    _session_header_ctx.set(None)


def merge_session_header(
    headers: Optional[Dict[str, str]], value: Optional[str]
) -> Dict[str, str]:
    """Return ``headers`` with ``X-Session-Id`` merged in.

    Never overwrites an explicit ``X-Session-Id`` (case-insensitive) —
    a caller that sets it deliberately wins. Never mutates the input
    dict. ``value`` falsy or ``headers``-is-a-string edge → input passed
    through (as a copy when we merged nothing, to keep this side-effect
    free).
    """
    if not value or not isinstance(headers, dict):
        # No value → nothing to add. ``headers=None`` is the normal
        # absent-headers case, not hostile: start a fresh dict. Any other
        # non-dict garbage cannot be merged into — degrade to empty rather
        # than raise into the call path.
        if not value:
            return dict(headers) if isinstance(headers, dict) else {}
        if headers is None:
            return {SESSION_HEADER_NAME: value}
        return {}
    merged = dict(headers)
    for key in merged:
        if isinstance(key, str) and key.lower() == _SESSION_HEADER_NAME_LOWER:
            return merged
    merged[SESSION_HEADER_NAME] = value
    return merged
