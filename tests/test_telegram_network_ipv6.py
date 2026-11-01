"""Tests for IPv6 support in the Telegram network fallback helpers.

Regression context: on networks where every Telegram IPv4 range is firewalled
(connection refused/RST), the fallback transport previously discarded all
IPv6 endpoints, leaving the gateway with no reachable path to api.telegram.org.
"""

import asyncio
import socket

import httpx

from gateway.platforms import telegram_network as tn


def test_normalize_accepts_global_ipv6():
    assert tn._normalize_fallback_ips(["2001:67c:4e8:f004::9"]) == [
        "2001:67c:4e8:f004::9"
    ]


def test_normalize_still_accepts_ipv4():
    assert tn._normalize_fallback_ips(["149.154.167.220"]) == ["149.154.167.220"]


def test_normalize_rejects_ipv6_local_addresses():
    assert tn._normalize_fallback_ips(["::1", "fe80::1", "fd00::1", "::"]) == []


def test_normalize_rejects_private_ipv4_and_garbage():
    assert tn._normalize_fallback_ips(["192.168.1.1", "not-an-ip", ""]) == []


def test_transport_dedupes_compressed_ipv6_forms():
    tf = tn.TelegramFallbackTransport(
        ["2001:67c:4e8:f004::9", "2001:67c:4e8:f004:0:0:0:9"]
    )
    assert tf._fallback_ips == ["2001:67c:4e8:f004::9"]


def test_parse_fallback_ip_env_accepts_ipv6():
    ips = tn.parse_fallback_ip_env("149.154.167.220, 2001:67c:4e8:f004::9")
    assert ips == ["149.154.167.220", "2001:67c:4e8:f004::9"]


def test_seed_fallback_ips_are_valid_and_include_ipv6():
    for ip in tn._SEED_FALLBACK_IPS:
        assert tn._normalize_fallback_ips([ip]) == [ip]
    assert any(":" in ip for ip in tn._SEED_FALLBACK_IPS)


def test_resolve_system_dns_includes_ipv6(monkeypatch):
    def fake_getaddrinfo(host, port, family=0):
        if family == socket.AF_INET:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("149.154.167.220", port))
            ]
        if family == socket.AF_INET6:
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("2001:67c:4e8:f004::9", port, 0, 0),
                )
            ]
        return []

    monkeypatch.setattr(tn.socket, "getaddrinfo", fake_getaddrinfo)
    ips = tn._resolve_system_dns()
    assert "149.154.167.220" in ips
    assert "2001:67c:4e8:f004::9" in ips


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "Answer": [
                {"type": 28, "data": "2001:67c:4e8:f004::9"},
                {"type": 1, "data": "149.154.167.220"},
                {"type": 5, "data": "cname.example"},  # ignored (CNAME)
            ]
        }


class _FakeClient:
    async def get(self, *args, **kwargs):
        return _FakeResponse()


def test_query_doh_provider_accepts_aaaa_answers():
    provider = {
        "url": "https://dns.test/resolve",
        "params": {"name": tn._TELEGRAM_API_HOST, "type": "AAAA"},
        "headers": {},
    }
    ips = asyncio.run(tn._query_doh_provider(_FakeClient(), provider))
    assert ips == ["2001:67c:4e8:f004::9", "149.154.167.220"]


def test_rewrite_request_for_ip_ipv6_keeps_sni_and_host():
    req = httpx.Request("GET", "https://api.telegram.org/bot123/getMe")
    rewritten = tn._rewrite_request_for_ip(req, "2001:67c:4e8:f004::9")
    assert "[2001:67c:4e8:f004::9]" in str(rewritten.url)
    assert rewritten.headers["host"] == "api.telegram.org"
    assert rewritten.extensions["sni_hostname"] == "api.telegram.org"


def test_rewrite_request_for_ip_ipv4_shape_unchanged():
    req = httpx.Request("GET", "https://api.telegram.org/")
    rewritten = tn._rewrite_request_for_ip(req, "149.154.167.220")
    assert rewritten.headers["host"] == "api.telegram.org"
    assert rewritten.extensions["sni_hostname"] == "api.telegram.org"


def test_fallback_transport_uses_ipv6_fallback_when_primary_fails():
    v6 = "2001:67c:4e8:f004::9"
    tf = tn.TelegramFallbackTransport([v6])
    sentinel = object()
    seen_urls = []

    class PrimaryFail(httpx.AsyncBaseTransport):
        def handle_async_request(self, request):
            raise httpx.ConnectError("all ipv4 blocked")

    class FakeOk(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            seen_urls.append(str(request.url))
            return sentinel

    tf._primary = PrimaryFail()
    tf._fallbacks = {v6: FakeOk()}

    resp = asyncio.run(
        tf.handle_async_request(httpx.Request("GET", "https://api.telegram.org/"))
    )
    assert resp is sentinel
    assert seen_urls and "[2001:67c:4e8:f004::9]" in seen_urls[0]
    assert tf._sticky_ip == v6
