#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Web4 Contributors
#
# Hardbound PolicyService Client
# https://github.com/dp-web4/web4

"""
Client for the Hardbound PolicyService server.

Provides server-side policy evaluation with graceful fallback to local
evaluation when the server is unreachable. The server URL is configured via:

1. Environment variable HARDBOUND_SERVER_URL
2. Config file ~/.web4/hardbound.json  {"server_url": "..."}
3. Default: http://localhost:9400

All methods are designed to fail gracefully — a network error never
blocks a tool call. The server is an enhancement, not a requirement.
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Connection timeout for server calls (seconds).
# Keep short — this runs in the pre_tool_use hot path.
_CONNECT_TIMEOUT = 2.0

# Cache the resolved URL and registration state per-process
_server_url = None
_plugin_id = None
_lct_id = None


def _get_server_url() -> str:
    """Resolve Hardbound server URL from env, config, or default."""
    global _server_url
    if _server_url is not None:
        return _server_url

    # 1. Environment variable
    url = os.environ.get("HARDBOUND_SERVER_URL")
    if url:
        _server_url = url.rstrip("/")
        return _server_url

    # 2. Config file
    config_file = Path.home() / ".web4" / "hardbound.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            url = config.get("server_url")
            if url:
                _server_url = url.rstrip("/")
                return _server_url
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Default
    _server_url = "http://localhost:9400"
    return _server_url


def _post(path: str, payload: dict, timeout: float = _CONNECT_TIMEOUT) -> dict:
    """
    POST JSON to the Hardbound server.

    Returns parsed response dict on success.
    Raises on any error (caller decides how to handle).
    """
    url = f"{_get_server_url()}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    resp = urlopen(req, timeout=timeout)
    body = resp.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def is_server_available() -> bool:
    """Quick check if the Hardbound server is reachable."""
    try:
        _post("/api/v1/plugin/heartbeat", {"plugin_id": _plugin_id or "probe"}, timeout=1.0)
        return True
    except Exception:
        return False


# ── Registration ─────────────────────────────────────────────────────

def register_plugin(session_id: str, session_token: dict) -> dict | None:
    """
    Register this plugin instance with the Hardbound server.

    Called once at session start.  Returns the server's registration
    response {plugin_id, lct_id, trust_ceiling} or None on failure.
    """
    global _plugin_id, _lct_id

    machine_hint = session_token.get("machine_hint", "unknown")
    payload = {
        "plugin_type": "web4-governance",
        "session_id": session_id,
        "token_id": session_token.get("token_id", ""),
        "machine_hint": machine_hint,
        "registered_at": datetime.now(timezone.utc).isoformat() + "Z",
    }

    try:
        resp = _post("/api/v1/plugin/register", payload, timeout=3.0)
        _plugin_id = resp.get("plugin_id")
        _lct_id = resp.get("lct_id")
        return resp
    except Exception:
        return None


def send_heartbeat() -> bool:
    """Send keep-alive heartbeat to server. Returns True on success."""
    if not _plugin_id:
        return False
    try:
        _post("/api/v1/plugin/heartbeat", {
            "plugin_id": _plugin_id,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }, timeout=1.5)
        return True
    except Exception:
        return False


# ── Policy Evaluation ────────────────────────────────────────────────

def evaluate_policy(
    tool_name: str,
    category: str,
    target: str,
    session: dict,
    full_command: str | None = None,
) -> tuple:
    """
    Ask the Hardbound server for a policy decision.

    Args:
        tool_name:    e.g. "Bash", "Write"
        category:     e.g. "command", "file_write"
        target:       primary target of the operation
        session:      session dict (provides actor LCT, role context)
        full_command: for Bash, the complete command string

    Returns:
        (decision, eval_dict) where decision is "allow"/"deny"/"escalate"
        and eval_dict has server metadata, **or** (None, None) if the
        server could not be reached (caller should fall back to local).
    """
    actor_lct = session.get("token", {}).get("token_id", "")
    policy_entity_id = session.get("policy_entity_id")

    payload = {
        "actor_lct": actor_lct,
        "action_type": category,
        "target": target,
        "parameters": {
            "tool_name": tool_name,
            "full_command": full_command,
        },
        "role_context": {
            "session_id": session.get("session_id", ""),
            "policy_entity_id": policy_entity_id,
            "action_index": session.get("action_count", 0),
            "plugin_id": _plugin_id,
        },
    }

    try:
        resp = _post("/api/v1/policy/evaluate", payload)
    except Exception:
        # Server unreachable — signal caller to fall back
        return None, None

    # Map server response to hook format
    decision = resp.get("decision", "allow")
    # Treat "escalate" as "allow" at the plugin level — the server may
    # surface escalation to a human dashboard, but the plugin should
    # not block on it.
    if decision == "escalate":
        decision = "allow"

    eval_dict = {
        "decision": decision,
        "rule_id": resp.get("rule_id"),
        "rule_name": resp.get("rule_name"),
        "reason": resp.get("reason"),
        "enforced": resp.get("enforced", True),
        "constraints": resp.get("constraints"),
        "request_id": resp.get("request_id"),
        "server_signed": resp.get("signature") is not None,
        "source": "hardbound-server",
    }

    return decision, eval_dict


# ── Outcome Reporting ────────────────────────────────────────────────

def report_outcome(
    request_id: str | None,
    success: bool,
    result_hash: str,
) -> bool:
    """
    Report action outcome to the Hardbound server.

    Called from post_tool_use after each tool completes.
    Returns True on success, False on failure (non-fatal).
    """
    if not request_id:
        return False

    payload = {
        "request_id": request_id,
        "success": success,
        "result_hash": result_hash,
        "reported_at": datetime.now(timezone.utc).isoformat() + "Z",
        "plugin_id": _plugin_id,
    }

    try:
        _post("/api/v1/policy/outcome", payload, timeout=1.5)
        return True
    except Exception:
        return False
