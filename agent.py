"""
Minimal LiveKit voice agent. Follows official examples only.
- basic_agent: https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
- mcp-agent: https://github.com/livekit/agents/blob/main/examples/voice_agents/mcp/mcp-agent.py
- nvidia_test: https://github.com/livekit/agents/blob/main/examples/voice_agents/nvidia_test.py
- flush_llm_node: https://github.com/livekit/agents/blob/main/examples/voice_agents/flush_llm_node.py
- Session/docs: https://docs.livekit.io/agents/build/sessions/
- Voice: no numbered/bullet lists (flowing sentences only) so TTS matches console on web.
- End call: agent has end_call tool; use when user says goodbye or wants to hang up.
- Inactivity: optional "are you there?" prompt and max call duration from room metadata.
"""
import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    FlushSentinel,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    RunContext,
    cli,
    inference,
    llm,
    metrics,
    room_io,
)
from livekit.agents.metrics import EOUMetrics, LLMMetrics, TTSMetrics

try:
    from livekit.agents import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip
except ImportError:
    AudioConfig = None
    BackgroundAudioPlayer = None
    BuiltinAudioClip = None
from livekit.agents.beta.tools.end_call import EndCallTool
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, silero
from livekit import rtc
from livekit.plugins.turn_detector.multilingual import MultilingualModel

try:
    from livekit.plugins import elevenlabs
except Exception:
    elevenlabs = None

try:
    from livekit.plugins import nvidia
except Exception:
    nvidia = None

load_dotenv()
logger = logging.getLogger("agent")

LIVEKIT_AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "").strip()
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "").strip()

def _mcp_servers():
    """MCP servers if URL set and livekit-agents[mcp] installed (see mcp-agent.py example). Skip localhost unless you run an MCP server."""
    if not MCP_SERVER_URL:
        return []
    url_lower = MCP_SERVER_URL.strip().lower()
    if "localhost" in url_lower or "127.0.0.1" in url_lower:
        logger.debug("MCP_SERVER_URL is localhost; skipping MCP to avoid connection errors when no server is running")
        return []
    try:
        from livekit.agents import mcp
        return [mcp.MCPServerHTTP(url=MCP_SERVER_URL)]
    except ImportError:
        logger.warning("MCP_SERVER_URL set but livekit-agents[mcp] not installed; pip install 'livekit-agents[mcp]'")
        return []


def _inbound_config_from_ctx(ctx: JobContext) -> dict | None:
    """Inbound config from API response, stored in proc.userdata so session uses it instead of room metadata."""
    proc = getattr(ctx, "proc", None)
    ud = getattr(proc, "userdata", None) if proc else None
    if not isinstance(ud, dict):
        return None
    return ud.get("inbound_config")


def _call_id_from_room(ctx: JobContext) -> str | None:
    """Read call id from room metadata or from inbound_config (API response)."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        cid = cfg.get("callId")
        if isinstance(cid, str) and cid.strip():
            return cid.strip()
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        call = data.get("call") or {}
        return call.get("id") if isinstance(call.get("id"), str) else None
    except Exception:
        return None


def _instructions_from_room(ctx: JobContext) -> str | None:
    """Read prompt from inbound_config (API response) or room metadata."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        prompt = cfg.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        prompt = (data.get("agent") or {}).get("prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None
    except Exception:
        return None


def _welcome_mode_from_room(ctx: JobContext) -> str:
    """Read welcome.mode from inbound_config or room metadata."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        w = cfg.get("welcome") or {}
        return "user" if w.get("mode") == "user" else "ai"
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return "ai"
    try:
        data = json.loads(raw)
        mode = (data.get("welcome") or {}).get("mode")
        return "user" if mode == "user" else "ai"
    except Exception:
        return "ai"


def _voice_from_room(ctx: JobContext) -> dict:
    """Read agent.voice from inbound_config or room metadata so dashboard voice selection is used."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        v = cfg.get("voice")
        return v if isinstance(v, dict) else {}
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        v = (data.get("agent") or {}).get("voice")
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _enabled_tools_from_room(ctx: JobContext) -> list[str]:
    """Read agent.enabledTools from inbound_config or room metadata (dashboard Tools tab)."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        t = cfg.get("enabledTools")
        if isinstance(t, list):
            return [str(x).strip() for x in t if isinstance(x, str) and x.strip()]
        return ["end_call"]
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return ["end_call"]
    try:
        data = json.loads(raw)
        t = (data.get("agent") or {}).get("enabledTools")
        if isinstance(t, list):
            return [str(x).strip() for x in t if isinstance(x, str) and x.strip()]
        return ["end_call"]
    except Exception:
        return ["end_call"]


def _backchannel_enabled_from_room(ctx: JobContext) -> bool:
    """Read agent.backchannelEnabled from inbound_config or room metadata."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        return bool(cfg.get("backchannelEnabled"))
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return False
    try:
        data = json.loads(raw)
        agent = data.get("agent") or {}
        return bool(agent.get("backchannelEnabled"))
    except Exception:
        return False


def _call_options_from_room(ctx: JobContext) -> dict:
    """Read call options from agent.callOptions or agent.maxCallSeconds (server already sends maxCallSeconds)."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        agent = data.get("agent") or {}
        opts = agent.get("callOptions") or {}
        max_sec = _positive_float(agent.get("maxCallSeconds"), 0)
        max_min_from_sec = max_sec / 60.0 if max_sec > 0 else 0
        max_min = _positive_float(opts.get("maxCallDurationMinutes"), max_min_from_sec)
        if max_min <= 0 and max_min_from_sec > 0:
            max_min = max_min_from_sec
        return {
            "max_call_duration_minutes": max_min,
            "inactivity_check_seconds": _positive_float(opts.get("inactivityCheckSeconds"), 0),
            "inactivity_prompt": str(opts.get("inactivityPrompt") or "Are you still there?").strip() or "Are you still there?",
            "end_call_after_inactivity_seconds": _positive_float(opts.get("endCallAfterInactivitySeconds"), 0),
        }
    except Exception:
        return {}


def _background_audio_from_room(ctx: JobContext) -> dict:
    """Read agent.backgroundAudio from room metadata (preset, ambientVolume)."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return {"preset": "none", "ambientVolume": 0.7}
    try:
        data = json.loads(raw)
        bg = (data.get("agent") or {}).get("backgroundAudio") or {}
        preset = str(bg.get("preset") or "none").strip().lower()
        if preset not in ("office", "keyboard", "office1", "office2"):
            preset = "none"
        vol = float(bg.get("ambientVolume", 0.7))
        vol = max(0.0, min(1.0, vol))
        return {"preset": preset, "ambientVolume": vol}
    except Exception:
        return {"preset": "none", "ambientVolume": 0.7}


def _positive_float(val, default: float) -> float:
    try:
        v = float(val) if val is not None else default
        return max(0.0, v)
    except (TypeError, ValueError):
        return default


def _call_settings_from_room(ctx: JobContext) -> dict:
    """Read agent.callSettings from inbound_config or room metadata."""
    cfg = _inbound_config_from_ctx(ctx)
    if cfg is not None:
        cs = cfg.get("callSettings")
        return cs if isinstance(cs, dict) else {}
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        cs = (data.get("agent") or {}).get("callSettings") or {}
        return cs if isinstance(cs, dict) else {}
    except Exception:
        return {}


def _fallback_voice_from_room(ctx: JobContext) -> dict | None:
    """Read agent.fallbackVoice from room metadata (provider, voiceId, model)."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        fv = (data.get("agent") or {}).get("fallbackVoice")
        return fv if isinstance(fv, dict) and fv else None
    except Exception:
        return None


def _post_call_extraction_from_room(ctx: JobContext) -> tuple[list, str]:
    """Read agent.postCallDataExtraction and postCallExtractionModel from room metadata."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return [], ""
    try:
        data = json.loads(raw)
        agent = data.get("agent") or {}
        items = agent.get("postCallDataExtraction")
        model = str(agent.get("postCallExtractionModel") or "").strip()
        return (items if isinstance(items, list) else []), model
    except Exception:
        return [], ""


def _build_stt(ctx: JobContext):
    """STT from room metadata: use NVIDIA when voice.provider is nvidia, else Deepgram."""
    voice_cfg = _voice_from_room(ctx)
    provider = str(voice_cfg.get("provider") or "").strip().lower()
    if provider == "nvidia" and nvidia:
        lang = str(voice_cfg.get("languageCode") or "en-US").strip() or "en-US"
        return nvidia.STT(language_code=lang)
    return inference.STT("deepgram/nova-3", language="multi")


def _build_tts(ctx: JobContext):
    """TTS from room metadata voice (provider, model, voiceId). callSettings/fallbackVoice/postCallDataExtraction read from metadata for future use."""
    voice_cfg = _voice_from_room(ctx)
    provider = str(voice_cfg.get("provider") or "").strip().lower()
    model = str(voice_cfg.get("model") or "").strip() or "sonic-3"
    voice_id = str(voice_cfg.get("voiceId") or "").strip() or "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

    if provider == "nvidia" and nvidia:
        nvidia_voice = voice_id if voice_id and not voice_id.startswith("96") else "Magpie-Multilingual.EN-US.Leo"
        lang = str(voice_cfg.get("languageCode") or "en-US").strip() or "en-US"
        return nvidia.TTS(voice=nvidia_voice, language_code=lang)
    if provider == "elevenlabs" and elevenlabs and voice_id:
        return elevenlabs.TTS(
            voice_id=voice_id,
            model=model or "eleven_turbo_v2_5",
        )
    # speed=1.0 for more stable audio (less breaking/artifacts on phone)
    try:
        return cartesia.TTS(
            model=model or "sonic-3",
            voice=voice_id,
            text_pacing=False,
            sample_rate=48000,
            speed=1.0,
        )
    except TypeError:
        try:
            return cartesia.TTS(model=model or "sonic-3", voice=voice_id, text_pacing=False, speed=1.0)
        except TypeError:
            return cartesia.TTS(model=model or "sonic-3", voice=voice_id, text_pacing=False)


class VoiceAgent(Agent):
    """Single agent with instructions and tools from dashboard (end_call, lookup_weather).
    Uses flush llm_node: when a tool is invoked without prior text, says a quick filler and flushes to TTS.
    """

    def __init__(
        self,
        instructions: str,
        speak_first: bool = True,
        tools: list | None = None,
        inbound_config: dict | None = None,
    ) -> None:
        tools = tools or [EndCallTool()]
        super().__init__(instructions=instructions, tools=tools)
        self._speak_first = speak_first
        self._inbound_config = inbound_config

    async def on_enter(self):
        if not self._speak_first:
            return
        # Give client time to calibrate AEC (basic_agent.py)
        self.session.generate_reply(allow_interruptions=False)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | FlushSentinel]:
        """Override to flush a quick filler to TTS when a tool is called (e.g. lookup_weather)."""
        called_tools: list[llm.FunctionToolCall] = []
        has_text_message = False
        async for chunk in Agent.default.llm_node(
            agent=self,
            chat_ctx=chat_ctx,
            tools=tools,
            model_settings=model_settings,
        ):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta:
                if chunk.delta.content:
                    has_text_message = True
                if chunk.delta.tool_calls:
                    called_tools.extend(chunk.delta.tool_calls)
            yield chunk
        tool_names = [t.name for t in called_tools]
        if not has_text_message and tool_names:
            filler = "One moment while I look that up. "
            logger.info("Flush filler for tool(s): %s", tool_names)
            yield filler
            yield FlushSentinel()

    def _room_metadata(self, context: RunContext) -> dict:
        """Read room metadata (call.to, agent.toolConfigs, etc.) for use in tools. Uses inbound_config when set."""
        cfg = getattr(self, "_inbound_config", None)
        if isinstance(cfg, dict):
            return {
                "call": {"id": cfg.get("callId"), "to": "unknown"},
                "agent": {"toolConfigs": cfg.get("toolConfigs") or {}},
            }
        raw = getattr(getattr(context, "room", None), "metadata", None) or ""
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather. Provide location; do not ask for lat/long."""
        logger.info("lookup_weather: %s", location)
        entries = getattr(self, "transcript_entries", None)
        tid = uuid.uuid4().hex[:16]
        inp = {"location": location, "latitude": latitude, "longitude": longitude}
        if entries is not None:
            entries.append({"kind": "tool_invocation", "toolCallId": tid, "toolName": "lookup_weather", "input": inp})
        result_str = "Sunny, 70°F."
        if entries is not None:
            entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "lookup_weather", "result": {"response": result_str}})
        return result_str

    @function_tool
    async def transfer_call(
        self, context: RunContext, execution_message: str = "Connecting you now."
    ) -> str:
        """Transfer the call to another number (e.g. live agent). Use when the user asks to be transferred. Provide a brief execution_message to say while transferring."""
        entries = getattr(self, "transcript_entries", None)
        meta = self._room_metadata(context)
        call = meta.get("call") or {}
        to_dest = call.get("to") or ""
        is_web = str(to_dest).strip().lower() == "webtest"
        agent_cfg = meta.get("agent") or {}
        tool_configs = agent_cfg.get("toolConfigs") or {}
        transfer_cfg = tool_configs.get("transfer_call") or {}
        transfer_to = (transfer_cfg.get("transferTo") or "").strip() or None

        tid = uuid.uuid4().hex[:16]
        inp = {"execution_message": execution_message}
        if entries is not None:
            entries.append({"kind": "tool_invocation", "toolCallId": tid, "toolName": "transfer_call", "input": inp})

        if is_web:
            msg = (
                "Transfer is not available on web calls. Please inform the customer and offer to help them "
                "directly or take a callback number."
            )
            result = {"status": "unavailable", "message": msg}
            if entries is not None:
                entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
            return msg

        if not transfer_to:
            msg = "No transfer number configured. Please inform the customer and offer to help them directly or take a callback number."
            result = {"status": "not_configured", "message": msg}
            if entries is not None:
                entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
            return msg

        call_id = call.get("id")
        base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
        secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
        if not call_id or not base_url or not secret:
            msg = "Transfer service unavailable. Please inform the customer and offer to try again or help them directly."
            result = {"status": "error", "message": msg}
            if entries is not None:
                entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
            return msg

        try:
            import requests
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{base_url}/api/internal/calls/{call_id}/transfer",
                    json={"transferTo": transfer_to},
                    headers={"x-agent-secret": secret, "content-type": "application/json"},
                    timeout=15,
                ),
            )
            if resp.status_code == 200:
                result = {"status": "transferred", "execution_message": execution_message}
                if entries is not None:
                    entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
                return "Transfer initiated. Please inform the customer they are being connected."
            err = resp.json() if resp.text else {}
            msg = err.get("message") or err.get("error") or resp.text or "Transfer failed."
            result = {"status": "failed", "message": msg}
            if entries is not None:
                entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
            return f"Transfer failed: {msg}. Please inform the customer that the transfer did not go through and offer to try again or assist them directly."
        except Exception as e:
            msg = str(e)
            result = {"status": "error", "message": msg}
            if entries is not None:
                entries.append({"kind": "tool_result", "toolCallId": tid, "toolName": "transfer_call", "result": result})
            return f"Transfer failed. Please inform the customer that the transfer did not go through and offer to try again or assist them directly."


server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm VAD (basic_agent.py)."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _room_has_agent_config(ctx: JobContext) -> bool:
    """True if room metadata already has agent.prompt (e.g. webtest or outbound, or inbound already started)."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return False
    try:
        data = json.loads(raw)
        prompt = (data.get("agent") or {}).get("prompt")
        return bool(isinstance(prompt, str) and prompt.strip())
    except Exception:
        return False


def _get_inbound_to_from(ctx: JobContext) -> tuple[str | None, str | None]:
    """
    Get (to, from) for inbound/start. Prefer LiveKit SIP participant attributes (correct for any dispatch rule):
    - sip.trunkPhoneNumber = number that was dialed (to)
    - sip.phoneNumber = caller (from)
    Fallback: parse room name (works if using Callee dispatch rule where room name contains the called number).
    """
    # 1) From SIP participant attributes (recommended by LiveKit docs) — works for Individual and Callee rules.
    try:
        kind_sip = getattr(rtc.ParticipantKind, "PARTICIPANT_KIND_SIP", None)
        for p in getattr(ctx.room, "remote_participants", {}).values():
            if getattr(p, "kind", None) != kind_sip:
                continue
            attrs = getattr(p, "attributes", None) or {}
            to_num = (attrs.get("sip.trunkPhoneNumber") or "").strip()
            from_num = (attrs.get("sip.phoneNumber") or "").strip()
            if to_num:
                return (to_num, from_num)
    except Exception as e:
        logger.debug("Reading SIP attributes: %s", e)
    # 2) Room name: with Callee rule room can be "number-+15551234567" or "call-+15551234567-abc"; with Individual it's caller number.
    import re
    room_name = getattr(ctx.room, "name", None) or ""
    parts = re.findall(r"\+\d{10,15}", room_name)
    if len(parts) >= 2:
        return (parts[0], parts[1])
    if len(parts) == 1:
        # Callee-style "call-+NUMBER" → that's the dialed number (to). Individual-style has caller number only (we'd need trunk from attributes).
        return (parts[0], "")
    return (None, None)


async def _ensure_inbound_config(ctx: JobContext) -> dict | None:
    """
    For inbound telephony: if room has no agent config, call the server's inbound/start
    and return the response so we use it directly (no reliance on room metadata propagation).
    Returns the JSON response dict on success, None otherwise.
    """
    if _room_has_agent_config(ctx):
        return None
    base_url = (os.environ.get("SERVER_BASE_URL") or os.environ.get("PUBLIC_API_BASE_URL") or "").strip().rstrip("/")
    secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
    if not base_url or not secret:
        logger.debug("Inbound config: skip (no SERVER_BASE_URL or AGENT_SHARED_SECRET)")
        return None
    to, from_ = _get_inbound_to_from(ctx)
    if not to:
        logger.debug("Inbound config: skip (could not get 'to' from room name or participants)")
        return None
    room_name = getattr(ctx.room, "name", None) or ""
    if not room_name:
        return None
    url = f"{base_url}/api/internal/telephony/inbound/start"
    try:
        import requests
        body = {"roomName": room_name, "to": to, "from": from_ or ""}
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                url,
                json=body,
                headers={"x-agent-secret": secret, "content-type": "application/json"},
                timeout=10,
            ),
        )
        if resp.status_code in (200, 201):
            data = resp.json() if resp.text else {}
            logger.info("Inbound config: got agent config for %s (using response directly)", room_name)
            return data
        logger.warning("Inbound config: POST %s returned %s %s", url, resp.status_code, (resp.text or "")[:200])
        return None
    except Exception as e:
        logger.warning("Inbound config: failed to call inbound/start: %s", e)
        return None


async def _run_session(ctx: JobContext, inbound_config: dict | None = None):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Connect first so we can read SIP participant attributes (sip.trunkPhoneNumber, sip.phoneNumber) for inbound/start.
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    # Give SIP participant a moment to appear in remote_participants so we can read sip.trunkPhoneNumber / sip.phoneNumber.
    if inbound_config is None and not _room_has_agent_config(ctx):
        await asyncio.sleep(0.5)
    # For inbound telephony: fetch agent config from our API using (to, from) from SIP attributes. Use response directly.
    if inbound_config is None and not _room_has_agent_config(ctx):
        inbound_config = await _ensure_inbound_config(ctx)
    if inbound_config is not None:
        ud = getattr(ctx.proc, "userdata", None)
        if isinstance(ud, dict):
            ud["inbound_config"] = inbound_config

    base = _instructions_from_room(ctx) or (
        "You are a helpful voice assistant. Keep responses concise. "
        "Do not use emojis or markdown. Speak naturally for TTS."
    )

    voice_rules = (
        "VOICE OUTPUT: Reply in short, flowing sentences. Do not use numbered lists (1. 2. 3.) "
        "or bullet points; they cause long pauses when spoken. Say the same content in plain prose. "
        "Use natural, conversational language as in a real phone call—avoid formal or written-style "
        "phrasing unless explicitly instructed."
        "Generate speech in full continuous sentences without micro-pauses between phrases; "
        "ensure stable prosody and consistent volume across the full utterance."
    )
    # End call: document the tool so the LLM (and you in the dashboard prompt) know when to use it.
    end_call_rule = (
        "END CALL: You have an end_call tool. When the user says goodbye, wants to hang up, or "
        "is done with the conversation, call end_call. Say a brief goodbye first, then call the tool."
    )
    enabled = _enabled_tools_from_room(ctx)
    backchannel_rule = ""
    if _backchannel_enabled_from_room(ctx):
        backchannel_rule = (
            "\n\nBACKCHANNELING (important): When the user is speaking or has just paused mid-thought "
            "(e.g. ended with \"and\", \"so\", \"but\", or an incomplete sentence), respond with ONLY a very short "
            "listener cue: \"yeah\", \"uh-huh\", \"mm-hmm\", \"right\", \"mhm\", or \"I see\". Do this DURING the "
            "conversation whenever the user is sharing a story or long message—offer one short backchannel, then "
            "let them continue. Keep your full replies for when they ask a direct question or finish a complete thought."
        )
    voicemail_rule = ""
    call_settings = _call_settings_from_room(ctx)
    if call_settings.get("voicemailDetectionEnabled"):
        if call_settings.get("voicemailResponse") == "hang_up":
            voicemail_rule = "\n\nVOICEMAIL: If you are told or detect that the call reached voicemail, say nothing further and call end_call immediately."
        elif call_settings.get("voicemailResponse") == "leave_message":
            msg_type = call_settings.get("voicemailMessageType") or "prompt"
            msg = (call_settings.get("voicemailStaticMessage") or call_settings.get("voicemailPrompt") or "").strip()
            if msg:
                voicemail_rule = f"\n\nVOICEMAIL: If you are told or detect that the call reached voicemail, deliver this message once then call end_call: {msg[:500]}"
            else:
                voicemail_rule = "\n\nVOICEMAIL: If you are told or detect that the call reached voicemail, leave a brief professional voicemail then call end_call."
    transfer_call_rule = ""
    if "transfer_call" in enabled:
        transfer_call_rule = (
            "\n\nTRANSFER: You have a transfer_call tool. When the user asks to speak to a person or be transferred, "
            "use it. If the tool returns that transfer is unavailable or failed, tell the customer clearly (e.g. "
            "\"Sorry, I couldn't transfer you\" or \"Transfer isn't available on this call\") and offer to help them "
            "directly or take a callback. Do NOT use end_call when transfer fails—only use end_call when the user "
            "says goodbye or is done with the conversation."
        )
    instructions = f"{base}\n\n{voice_rules}\n\n{end_call_rule}{transfer_call_rule}{backchannel_rule}{voicemail_rule}"
    speak_first = _welcome_mode_from_room(ctx) != "user"

    # Collect transcript for post-call extraction (call_analyzed webhook)
    transcript_entries: list[dict] = []

    async def _post_end_call_to_server() -> None:
        """POST to RapidCall API so it can hang up the SIP leg, run extraction, and send call_ended/call_analyzed webhooks."""
        call_id = _call_id_from_room(ctx)
        base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
        secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
        if not call_id or not base_url or not secret:
            logger.debug("End call: not posting to server (missing call_id, SERVER_BASE_URL, or AGENT_SHARED_SECRET)")
            return
        url = f"{base_url}/api/internal/calls/{call_id}/end"
        try:
            import requests
            body: dict = {"outcome": "agent_hangup"}
            if transcript_entries:
                body["transcript"] = transcript_entries
                logger.info("End call: sending transcript with %d entries for call_analyzed", len(transcript_entries))
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json=body,
                    headers={"x-agent-secret": secret, "content-type": "application/json"},
                    timeout=15,
                ),
            )
            if resp.status_code >= 400:
                logger.warning("POST %s failed: %s %s", url, resp.status_code, resp.text[:200])
            else:
                logger.info("Posted end call to server for call %s (SIP hangup + webhooks)", call_id)
        except Exception as e:
            logger.warning("Failed to post end call to server: %s", e)

    def _append_end_call_tool_entries() -> None:
        """Append tool_invocation and tool_result for end_call so they appear in call history (Retell-style)."""
        call_id = _call_id_from_room(ctx)
        tid = uuid.uuid4().hex[:16]
        transcript_entries.append({
            "kind": "tool_invocation",
            "toolCallId": tid,
            "toolName": "end_call",
            "input": {"execution_message": "Call ended by agent."},
        })
        transcript_entries.append({
            "kind": "tool_result",
            "toolCallId": tid,
            "toolName": "end_call",
            "result": {"execution_message": "Call ended by agent."},
        })

    async def _on_end_call_tool_called(_ev) -> None:
        _append_end_call_tool_entries()
        await _post_end_call_to_server()

    end_call_tool = (
        EndCallTool(on_tool_called=_on_end_call_tool_called) if "end_call" in enabled else None
    )
    tool_list = [end_call_tool] if end_call_tool else []
    # lookup_weather / transfer_call are on VoiceAgent as @function_tool; add agent when enabled
    tts_obj = _build_tts(ctx)
    stt_obj = _build_stt(ctx)

    session = AgentSession(
        stt=stt_obj,
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts_obj,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        # When user interrupts and then speaks, respond to the NEW message instead of resuming the cut-off reply.
        resume_false_interruption=False,
        false_interruption_timeout=0.3,
        min_interruption_duration=0.2,
        mcp_servers=_mcp_servers(),
    )

    # Collect transcript for post-call extraction (call_analyzed webhook)
    try:
        from livekit.agents import ConversationItemAddedEvent

        def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
            item = getattr(ev, "item", None)
            if not item:
                return
            role = getattr(item, "role", None)
            text = (getattr(item, "text_content", None) or "").strip()
            if not text:
                return
            r = "user" if role == "user" else "agent"
            speaker = "User" if r == "user" else "Agent"
            transcript_entries.append({"speaker": speaker, "role": r, "text": text[:5000]})

        session.on("conversation_item_added", _on_conversation_item_added)
    except Exception as e:
        logger.debug("Could not register conversation_item_added for transcript: %s", e)

    # Sync transcript to server periodically so when user hangs up we have it (transcript + analysis)
    async def _sync_transcript_to_server() -> None:
        if not transcript_entries:
            return
        call_id = _call_id_from_room(ctx)
        base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
        secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
        if not call_id or not base_url or not secret:
            return
        url = f"{base_url}/api/internal/calls/{call_id}/transcript"
        try:
            import requests
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json={"transcript": list(transcript_entries)},
                    headers={"x-agent-secret": secret, "content-type": "application/json"},
                    timeout=5,
                ),
            )
            if resp.status_code >= 400:
                logger.debug("Transcript sync failed: %s %s", resp.status_code, resp.text[:100])
        except Exception as e:
            logger.debug("Transcript sync failed: %s", e)

    _transcript_sync_task: asyncio.Task | None = None

    async def _transcript_sync_loop() -> None:
        while True:
            await asyncio.sleep(10.0)
            await _sync_transcript_to_server()

    try:
        _transcript_sync_task = asyncio.create_task(_transcript_sync_loop())
        ctx.add_shutdown_callback(lambda: _transcript_sync_task.cancel() if _transcript_sync_task else None)
    except Exception as e:
        logger.debug("Could not start transcript sync task: %s", e)

    def _on_shutdown_push_transcript():
        """On shutdown (e.g. user hung up), push transcript once so call detail has full transcript."""
        if not transcript_entries:
            return
        call_id = _call_id_from_room(ctx)
        base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
        secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
        if not call_id or not base_url or not secret:
            return
        try:
            import requests
            requests.post(
                f"{base_url}/api/internal/calls/{call_id}/transcript",
                json={"transcript": list(transcript_entries)},
                headers={"x-agent-secret": secret, "content-type": "application/json"},
                timeout=5,
            )
            logger.info("Shutdown: pushed transcript (%d entries) for call %s", len(transcript_entries), call_id)
        except Exception as e:
            logger.warning("Shutdown: failed to push transcript: %s", e)
    ctx.add_shutdown_callback(_on_shutdown_push_transcript)

    # Subscribe to remote audio (needed when using SUBSCRIBE_NONE).
    def _subscribe_audio(participant):
        for pub in getattr(participant, "track_publications", {}).values():
            if getattr(pub, "kind", None) == rtc.TrackKind.KIND_AUDIO and not getattr(pub, "subscribed", False):
                pub.set_subscribed(True)

    ctx.room.on("participant_connected", _subscribe_audio)
    ctx.room.on("track_published", lambda pub, p: _subscribe_audio(p))
    for p in getattr(ctx.room, "remote_participants", {}).values():
        _subscribe_audio(p)

    usage_collector = metrics.UsageCollector()

    # Latency: EOU + LLM ttft + TTS ttfb (see https://docs.livekit.io/deploy/observability/data/)
    latency_eou_delays: list[float] = []
    latency_eou_transcription_delays: list[float] = []
    latency_llm_ttfts: list[float] = []
    latency_tts_ttfbs: list[float] = []
    _latency_posted: dict = {"done": False}

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        m = ev.metrics
        if isinstance(m, EOUMetrics):
            latency_eou_delays.append(m.end_of_utterance_delay)
            latency_eou_transcription_delays.append(m.transcription_delay)
        elif isinstance(m, LLMMetrics):
            latency_llm_ttfts.append(m.ttft)
        elif isinstance(m, TTSMetrics):
            latency_tts_ttfbs.append(m.ttfb)
        # Post latency as soon as we have one full turn so the UI gets it quickly.
        if (
            not _latency_posted["done"]
            and len(latency_eou_delays) >= 1
            and len(latency_llm_ttfts) >= 1
            and len(latency_tts_ttfbs) >= 1
        ):
            try:
                asyncio.get_running_loop().create_task(_post_latency_metrics())
            except RuntimeError:
                pass

    async def log_usage():
        logger.info("Usage: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    def _build_latency_payload():
        n_eou = len(latency_eou_delays)
        n_llm = len(latency_llm_ttfts)
        n_tts = len(latency_tts_ttfbs)
        if n_eou == 0 and n_llm == 0 and n_tts == 0:
            return None
        eou_avg = sum(latency_eou_delays) / n_eou if n_eou else 0.0
        eou_trans_avg = sum(latency_eou_transcription_delays) / n_eou if n_eou else 0.0
        llm_ttft_avg = sum(latency_llm_ttfts) / n_llm if n_llm else 0.0
        tts_ttfb_avg = sum(latency_tts_ttfbs) / n_tts if n_tts else 0.0
        agent_turn_ms = (eou_avg + llm_ttft_avg + tts_ttfb_avg) * 1000.0
        return {
            "latency": {
                "eou_end_ms_avg": round(eou_avg * 1000.0),
                "eou_transcription_ms_avg": round(eou_trans_avg * 1000.0),
                "llm_ttft_ms_avg": round(llm_ttft_avg * 1000.0),
                "agent_turn_latency_ms_avg": round(agent_turn_ms),
            }
        }

    async def _post_latency_metrics():
        if _latency_posted["done"]:
            return
        call_id = _call_id_from_room(ctx)
        base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
        secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
        if not call_id:
            logger.debug("Latency metrics: no call id in room metadata (metrics not posted)")
            return
        if not base_url:
            logger.warning("Latency metrics: SERVER_BASE_URL not set (metrics not posted)")
            return
        if not secret:
            logger.warning("Latency metrics: AGENT_SHARED_SECRET not set (metrics not posted)")
            return
        payload = _build_latency_payload()
        if not payload:
            return
        _latency_posted["done"] = True
        url = f"{base_url}/api/calls/{call_id}/metrics"
        try:
            import requests
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json=payload,
                    headers={"x-agent-secret": secret, "content-type": "application/json"},
                    timeout=10,
                ),
            )
            if resp.status_code >= 400:
                logger.warning("POST %s failed: %s %s", url, resp.status_code, resp.text[:200])
            else:
                logger.info("Posted latency metrics for call %s", call_id)
        except Exception as e:
            logger.warning("Failed to post latency metrics: %s", e)

    ctx.add_shutdown_callback(_post_latency_metrics)

    async def _periodic_latency_post():
        """Post latency every 20s so we have data even if shutdown doesn't run."""
        while True:
            await asyncio.sleep(20.0)
            if _latency_posted["done"]:
                continue
            payload = _build_latency_payload()
            if not payload:
                continue
            call_id = _call_id_from_room(ctx)
            base_url = (os.environ.get("SERVER_BASE_URL") or "").strip().rstrip("/")
            secret = (os.environ.get("AGENT_SHARED_SECRET") or "").strip()
            if not call_id or not base_url or not secret:
                continue
            _latency_posted["done"] = True
            url = f"{base_url}/api/calls/{call_id}/metrics"
            try:
                import requests
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        url,
                        json=payload,
                        headers={"x-agent-secret": secret, "content-type": "application/json"},
                        timeout=10,
                    ),
                )
                if resp.status_code >= 400:
                    logger.warning("POST %s (periodic) failed: %s", url, resp.status_code)
                else:
                    logger.info("Posted latency metrics (periodic) for call %s", call_id)
            except Exception as e:
                logger.warning("Periodic latency post failed: %s", e)

    _latency_task = asyncio.create_task(_periodic_latency_post())
    ctx.add_shutdown_callback(lambda: _latency_task.cancel() if _latency_task else None)

    # Inactivity and max call duration (from room metadata agent.callOptions)
    call_opts = _call_options_from_room(ctx)
    last_activity = {"t": time.monotonic()}
    session_start = time.monotonic()
    inactivity_prompted_at: float | None = None

    backchannel_enabled = _backchannel_enabled_from_room(ctx)
    last_backchannel_at = {"t": 0.0}
    BACKCHANNEL_INTERVAL = 4.0
    BACKCHANNEL_MIN_LEN = 12
    BACKCHANNEL_FINAL_MIN_LEN = 40
    BACKCHANNEL_PHRASES = ("Mm-hmm.", "Yeah.", "Uh-huh.", "Right.", "I see.")

    def _on_user_input(ev):
        is_final = getattr(ev, "is_final", getattr(ev, "final", True))
        transcript = (getattr(ev, "transcript", None) or "").strip()
        if is_final:
            last_activity["t"] = time.monotonic()
            if not backchannel_enabled or len(transcript) < BACKCHANNEL_FINAL_MIN_LEN:
                return
            now = time.monotonic()
            if now - last_backchannel_at["t"] < BACKCHANNEL_INTERVAL:
                return
            last_backchannel_at["t"] = now
            phrase = random.choice(BACKCHANNEL_PHRASES)
            try:
                logger.info("Backchannel (after final): %s", phrase)
                out = session.say(phrase)
                if asyncio.iscoroutine(out):
                    asyncio.get_running_loop().create_task(out)
            except Exception as e:
                logger.debug("Backchannel say failed: %s", e)
            return
        if not backchannel_enabled or len(transcript) < BACKCHANNEL_MIN_LEN:
            return
        now = time.monotonic()
        if now - last_backchannel_at["t"] < BACKCHANNEL_INTERVAL:
            return
        last_backchannel_at["t"] = now
        phrase = random.choice(BACKCHANNEL_PHRASES)
        try:
            logger.info("Backchannel (interim): %s", phrase)
            out = session.say(phrase)
            if asyncio.iscoroutine(out):
                asyncio.get_running_loop().create_task(out)
        except Exception as e:
            logger.debug("Backchannel say failed: %s", e)

    try:
        session.on("user_input_transcribed", _on_user_input)
    except Exception:
        pass

    async def _inactivity_and_max_duration_loop():
        nonlocal inactivity_prompted_at
        check_interval = 10.0
        max_dur = call_opts.get("max_call_duration_minutes") or 0
        inact_check = call_opts.get("inactivity_check_seconds") or 0
        inact_prompt = call_opts.get("inactivity_prompt") or "Are you still there?"
        end_after_inact = call_opts.get("end_call_after_inactivity_seconds") or 0
        if max_dur <= 0 and inact_check <= 0:
            return
        while True:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            elapsed_min = (now - session_start) / 60.0
            if max_dur > 0 and elapsed_min >= max_dur:
                logger.info("Max call duration reached (%.1f min), ending call", max_dur)
                await ctx.shutdown(reason="max_call_duration")
                return
            since_activity = now - last_activity["t"]
            if inact_check > 0 and since_activity >= inact_check:
                if inactivity_prompted_at is None:
                    logger.info("User inactive %.0fs, saying: %s", since_activity, inact_prompt[:50])
                    inactivity_prompted_at = now
                    try:
                        session.say(inact_prompt)
                    except Exception as e:
                        logger.warning("generate_reply for inactivity failed: %s", e)
                elif end_after_inact > 0 and (now - inactivity_prompted_at) >= end_after_inact:
                    logger.info("User still inactive after prompt, ending call")
                    await ctx.shutdown(reason="inactivity")
                    return
            else:
                inactivity_prompted_at = None

    loop_task: asyncio.Task | None = None
    if (call_opts.get("max_call_duration_minutes") or 0) > 0 or (call_opts.get("inactivity_check_seconds") or 0) > 0:
        loop_task = asyncio.create_task(_inactivity_and_max_duration_loop())
        ctx.add_shutdown_callback(lambda: loop_task.cancel() if loop_task else None)

    bg_audio_task: asyncio.Task | None = None
    if BackgroundAudioPlayer and AudioConfig and BuiltinAudioClip:
        bg = _background_audio_from_room(ctx)
        preset = bg.get("preset") or "none"
        vol = float(bg.get("ambientVolume", 0.7))
        if preset != "none" and vol > 0:
            try:
                if preset == "office":
                    ambient = AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=vol)
                elif preset == "keyboard":
                    ambient = AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=vol)
                elif preset == "office1":
                    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
                    path = os.path.join(audio_dir, "Office1.mp3")
                    ambient = AudioConfig(path, volume=vol)
                elif preset == "office2":
                    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
                    path = os.path.join(audio_dir, "Office2.mp3")
                    ambient = AudioConfig(path, volume=vol)
                else:
                    ambient = None
                if ambient is not None:
                    background_audio = BackgroundAudioPlayer(ambient_sound=ambient)
                    async def _start_bg_audio():
                        await asyncio.sleep(2.0)
                        try:
                            await background_audio.start(room=ctx.room, agent_session=session)
                        except Exception as e:
                            logger.warning("Background audio start failed: %s", e)
                    bg_audio_task = asyncio.create_task(_start_bg_audio())
                    ctx.add_shutdown_callback(lambda: bg_audio_task.cancel() if bg_audio_task else None)
            except Exception as e:
                logger.warning("Background audio setup failed: %s", e)

    agent = VoiceAgent(
        instructions=instructions,
        speak_first=speak_first,
        tools=tool_list,
        inbound_config=inbound_config,
    )
    agent.transcript_entries = transcript_entries
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(sample_rate=48000),
            audio_output=room_io.AudioOutputOptions(sample_rate=48000),
            text_output=room_io.TextOutputOptions(sync_transcription=True),
        ),
    )


if LIVEKIT_AGENT_NAME:
    @server.rtc_session(agent_name=LIVEKIT_AGENT_NAME)
    async def entrypoint(ctx: JobContext):
        await _run_session(ctx)
else:
    @server.rtc_session()
    async def entrypoint(ctx: JobContext):
        await _run_session(ctx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cli.run_app(server)
