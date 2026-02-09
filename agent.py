import asyncio
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    AutoSubscribe,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.agents import inference
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # type: ignore

# Krisp noise cancellation — off by default (matches official LiveKit docs).
# https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
# Only used for telephony calls; BVC on WebRTC audio can filter out user speech entirely.
from livekit.plugins import noise_cancellation

try:
    # Optional: only present if you add livekit-plugins-elevenlabs
    from livekit.plugins import elevenlabs  # type: ignore
except Exception:  # pragma: no cover
    elevenlabs = None  # type: ignore

load_dotenv()
logger = logging.getLogger("basic-agent")

# Optional explicit agent name. If set, this agent can be targeted by LiveKit Telephony dispatch rules / playground.
# IMPORTANT: When agent_name is set, LiveKit uses explicit dispatch (not automatic room assignment).
LIVEKIT_AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "").strip()

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _extract_prompt_from_room(ctx: JobContext) -> str | None:
    """
    The Node server sets room metadata like:
      {"agent": {"name": "...", "prompt": "..." }}
    """
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    prompt = (
        data.get("agent", {}).get("prompt")
        if isinstance(data, dict)
        else None
    )
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def _extract_welcome_from_room(ctx: JobContext) -> dict | None:
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    welcome = data.get("welcome") if isinstance(data, dict) else None
    return welcome if isinstance(welcome, dict) else None


def _extract_call_id_from_room(ctx: JobContext) -> str | None:
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    call = data.get("call") if isinstance(data, dict) else None
    call_id = call.get("id") if isinstance(call, dict) else None
    return call_id if isinstance(call_id, str) and call_id.strip() else None


def _extract_agent_name_from_room(ctx: JobContext) -> str | None:
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    name = data.get("agent", {}).get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def _extract_llm_model_from_room(ctx: JobContext) -> str | None:
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    m = data.get("agent", {}).get("llmModel") if isinstance(data, dict) else None
    return m if isinstance(m, str) and m.strip() else None


def _extract_knowledge_folder_ids_from_room(ctx: JobContext) -> list[str]:
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    v = data.get("agent", {}).get("knowledgeFolderIds") if isinstance(data, dict) else None
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out[:50]


# ── Background audio helpers ──────────────────────────────────────────
# Mapping of user-facing background audio option keys to BuiltinAudioClip + defaults.
BACKGROUND_AUDIO_OPTIONS: dict[str, dict] = {
    "office": {
        "label": "Office ambience",
        "ambient": BuiltinAudioClip.OFFICE_AMBIENCE,
        "ambient_volume": 0.7,
        "thinking": [BuiltinAudioClip.KEYBOARD_TYPING, BuiltinAudioClip.KEYBOARD_TYPING2],
        "thinking_volume": 0.7,
    },
    "keyboard": {
        "label": "Keyboard typing",
        "ambient": BuiltinAudioClip.KEYBOARD_TYPING,
        "ambient_volume": 0.6,
        "thinking": [BuiltinAudioClip.KEYBOARD_TYPING2],
        "thinking_volume": 0.7,
    },
    "none": {
        "label": "No background audio",
        "ambient": None,
        "ambient_volume": 0,
        "thinking": [],
        "thinking_volume": 0,
    },
}


def _extract_background_audio_from_room(ctx: JobContext) -> dict | None:
    """Extract background audio config from room metadata (agent.backgroundAudio)."""
    raw = getattr(ctx.room, "metadata", None)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    ba = data.get("agent", {}).get("backgroundAudio") if isinstance(data, dict) else None
    return ba if isinstance(ba, dict) else None


def _build_background_audio_player(bg_cfg: dict | None) -> BackgroundAudioPlayer | None:
    """Build a BackgroundAudioPlayer from a config dict like {"preset": "office", "ambientVolume": 0.7}."""
    if not bg_cfg:
        return None

    preset_key = str(bg_cfg.get("preset") or "").strip().lower()
    if not preset_key or preset_key == "none":
        return None

    preset = BACKGROUND_AUDIO_OPTIONS.get(preset_key)
    if not preset or preset.get("ambient") is None:
        return None

    ambient_vol = float(bg_cfg.get("ambientVolume", preset.get("ambient_volume", 0.7)))
    thinking_vol = float(bg_cfg.get("thinkingVolume", preset.get("thinking_volume", 0.7)))

    # Ambient sound
    ambient_sound = AudioConfig(preset["ambient"], volume=max(0.0, min(1.0, ambient_vol)))

    # Thinking sounds
    thinking_clips = preset.get("thinking") or []
    thinking_sound = [
        AudioConfig(clip, volume=max(0.0, min(1.0, thinking_vol)))
        for clip in thinking_clips
    ] if thinking_clips else None

    return BackgroundAudioPlayer(
        ambient_sound=ambient_sound,
        thinking_sound=thinking_sound,
    )


def _post_call_metrics(call_id: str, payload: dict) -> None:
    base = (
        os.environ.get("SERVER_BASE_URL", "").strip().rstrip("/")
        or os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    )
    if not base:
        return
    secret = os.environ.get("AGENT_SHARED_SECRET", "").strip()
    url = f"{base}/api/calls/{call_id}/metrics"
    body = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if secret:
        headers["x-agent-secret"] = secret
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    try:
        timeout_s = float(os.environ.get("AGENT_METRICS_TIMEOUT_S", "5") or "5")
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
    except (HTTPError, URLError) as e:
        logger.warning(f"Failed to post call metrics: {e}")


def _post_internal_json(path: str, payload: dict) -> dict | None:
    base = (
        os.environ.get("SERVER_BASE_URL", "").strip().rstrip("/")
        or os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    )
    if not base:
        return None
    secret = os.environ.get("AGENT_SHARED_SECRET", "").strip()
    if not secret:
        logger.warning("AGENT_SHARED_SECRET not set; cannot call internal API")
        return None
    url = f"{base}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        headers={"content-type": "application/json", "x-agent-secret": secret},
        method="POST",
    )
    try:
        timeout_s = float(os.environ.get("AGENT_INTERNAL_TIMEOUT_S", "10") or "10")
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except (HTTPError, URLError) as e:
        logger.warning(f"Internal API call failed {path}: {e}")
        return None


async def _post_call_metrics_with_retry(call_id: str, payload: dict, *, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            _post_call_metrics(call_id, payload)
            return
        except Exception as e:
            if attempt >= max_retries - 1:
                logger.error(f"Failed to post call metrics after {max_retries} attempts: {e}")
                return
            await asyncio.sleep(2 ** attempt)


async def _post_internal_json_with_retry(path: str, payload: dict, *, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = await asyncio.to_thread(_post_internal_json, path, payload)
            if resp is not None:
                return resp
        except Exception as e:
            if attempt >= max_retries - 1:
                logger.error(f"Internal API call failed after {max_retries} attempts {path}: {e}")
                return None
        await asyncio.sleep(2 ** attempt)
    return None


def _extract_sip_numbers(ctx: JobContext) -> tuple[str | None, str | None]:
    """
    Attempt to read SIP participant attributes provided by LiveKit Telephony.
    Docs: https://docs.livekit.io/reference/telephony/sip-participant/
    """
    try:
        parts = getattr(ctx.room, "remote_participants", None) or {}
        vals = parts.values() if hasattr(parts, "values") else []
    except Exception:
        vals = []

    trunk_number = None
    caller_number = None
    for p in vals:
        attrs = getattr(p, "attributes", None) or {}
        if not isinstance(attrs, dict):
            continue
        trunk_number = trunk_number or attrs.get("sip.trunkPhoneNumber")
        caller_number = caller_number or attrs.get("sip.phoneNumber")
        if trunk_number or caller_number:
            # keep scanning in case one is missing
            pass

    return trunk_number, caller_number


def _extract_sip_call_sid(ctx: JobContext) -> str | None:
    """
    Best-effort extraction of provider call id from LiveKit SIP participant attributes.
    Depending on the provider/bridge, this may or may not be present.
    """
    try:
        parts = getattr(ctx.room, "remote_participants", None) or {}
        vals = parts.values() if hasattr(parts, "values") else []
    except Exception:
        vals = []

    for p in vals:
        attrs = getattr(p, "attributes", None) or {}
        if not isinstance(attrs, dict):
            continue
        # Try common keys (case-insensitive)
        for k, v in attrs.items():
            if not isinstance(k, str):
                continue
            key = k.strip().lower()
            if key in ("sip.callid", "sip.call_id", "sip.callsid", "sip.twiliocallsid", "twilio.callsid"):
                vv = str(v or "").strip()
                if vv:
                    return vv
            if "callsid" in key or (key.startswith("sip.") and "call" in key and "id" in key):
                vv = str(v or "").strip()
                if vv:
                    return vv
    return None


async def _wait_for_sip_numbers(ctx: JobContext, *, timeout_s: float = 6.0) -> tuple[str | None, str | None]:
    """
    In telephony rooms, the SIP participant may join slightly after we connect.
    Retry briefly so we can reliably read sip.trunkPhoneNumber/sip.phoneNumber and fetch the real prompt.
    """
    deadline = asyncio.get_event_loop().time() + max(0.0, timeout_s)
    last = (None, None)
    attempt = 0
    while asyncio.get_event_loop().time() < deadline:
        attempt += 1
        last = _extract_sip_numbers(ctx)
        if last[0] or last[1]:
            return last
        await asyncio.sleep(0.4)
    logger.info(f"SIP attributes not found after {attempt} attempts; falling back to room metadata prompt")
    return last


class MyAgent(Agent):
    def __init__(
        self,
        *,
        extra_prompt: str | None = None,
        welcome: dict | None = None,
        call_id: str | None = None,
        kb_folder_ids: list[str] | None = None,
        speaker_name: str | None = None,
    ) -> None:
        style = (
            "You interact with users via voice. Keep responses concise and to the point. "
            "Do not use emojis, asterisks, markdown, or other special characters. "
            "Speak English unless the user requests otherwise. "
            "Speak naturally, with short pauses at commas and longer pauses at periods. "
            "Do not run words together. Do not read punctuation out loud."
        )

        kb_hint = ""
        if kb_folder_ids:
            kb_hint = (
                "KNOWLEDGE BASE:\n"
                "- You have access to a Knowledge Base via the kb_search tool.\n"
                "- When the user asks about documents, facts, policies, or anything that could be in the Knowledge Base, call kb_search first.\n"
                "- Use the returned excerpts as evidence in your answer.\n\n"
            )

        # IMPORTANT: Do not hardcode a persona beyond the configured name; the per-agent prompt should control behavior.
        if extra_prompt and extra_prompt.strip():
            instructions = (
                "CUSTOM AGENT PROMPT (highest priority):\n"
                f"{extra_prompt.strip()}\n\n"
                f"{kb_hint}"
                "ADDITIONAL STYLE CONSTRAINTS:\n"
                f"{style}"
            )
        else:
            instructions = (
                "You are a helpful voice assistant.\n\n"
                f"{kb_hint}"
                "ADDITIONAL STYLE CONSTRAINTS:\n"
                f"{style}"
            )

        super().__init__(instructions=instructions)
        self._welcome = welcome or {}
        self._call_id = call_id
        self._kb_folder_ids = kb_folder_ids or []

    async def on_enter(self):
        mode = str(self._welcome.get("mode") or "user")
        if mode == "user":
            # User speaks first: do nothing.
            return

        # AI speaks first
        delay = self._welcome.get("aiDelaySeconds", 0) or 0
        try:
            delay_f = float(delay)
        except Exception:
            delay_f = 0.0
        if delay_f > 0:
            await asyncio.sleep(min(max(delay_f, 0.0), 10.0))

        msg_mode = str(self._welcome.get("aiMessageMode") or "dynamic")
        if msg_mode == "custom":
            text = str(self._welcome.get("aiMessageText") or "").strip()
            if text:
                self.session.say(text, allow_interruptions=False, add_to_chat_ctx=True)
                return

        # Dynamic: generate a greeting using the agent prompt, but disable tools.
        self.session.generate_reply(
            instructions=(
                "Greet the user in one short sentence based on the CUSTOM AGENT PROMPT, "
                "then ask exactly one opening question. Do not call tools."
            ),
            allow_interruptions=False,
            tool_choice="none",
        )

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        logger.info(f"Looking up weather for {location} ({latitude},{longitude})")
        return "sunny with a temperature of 70 degrees."

    @function_tool
    async def get_current_datetime(self, context: RunContext) -> str:
        """Return the current local date/time. Only call when the user asks."""
        now = datetime.now().astimezone()
        tz = now.strftime("%z")
        tz_fmt = f"UTC{tz[:3]}:{tz[3:]}" if tz else "local time"
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({tz_fmt})")

    @function_tool
    async def kb_search(self, context: RunContext, query: str) -> str:
        """
        Search the workspace Knowledge Base within the folders linked to this agent.
        """
        q = str(query or "").strip()
        if not q:
            return "Missing query."
        if not self._call_id:
            return "Knowledge Base search is unavailable (missing call id)."
        if not self._kb_folder_ids:
            return "No Knowledge Base folders are connected to this agent."

        resp = await _post_internal_json_with_retry(
            "/api/internal/kb/search",
            {"callId": self._call_id, "query": q, "folderIds": self._kb_folder_ids, "limit": 5},
        )
        if not isinstance(resp, dict):
            return "Knowledge Base search failed."
        results = resp.get("results")
        if not isinstance(results, list) or len(results) == 0:
            return "No matches found in the connected Knowledge Base."

        lines: list[str] = []
        for r in results[:5]:
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or "Document").strip()
            excerpt = str(r.get("excerpt") or "").strip()
            if excerpt:
                lines.append(f"- {title}: {excerpt}")
            else:
                lines.append(f"- {title}")
        return "Matches:\n" + "\n".join(lines)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Reuse model clients across sessions to reduce "press Talk → agent ready" latency.
    # STT backend:
    # - livekit: LiveKit Inference Gateway (may 429 if your project has no inference quota)
    # - deepgram: direct Deepgram plugin (recommended when inference quota is exceeded)
    stt_backend = os.environ.get("STT_BACKEND", "deepgram").strip().lower()
    if stt_backend == "livekit":
        proc.userdata["stt"] = inference.STT(
            model=os.environ.get("LK_STT_MODEL", "auto"),
            extra_kwargs={"interim_results": True},
        )
        proc.userdata["stt_model_name"] = f"livekit/{os.environ.get('LK_STT_MODEL', 'auto')}"
    else:
        proc.userdata["stt"] = deepgram.STT(
            model=os.environ.get("DEEPGRAM_STT_MODEL", "nova-3"),
            language=os.environ.get("DEEPGRAM_LANGUAGE", "en-US"),
            interim_results=True,
            punctuate=True,
        )
        proc.userdata["stt_model_name"] = f"deepgram/{os.environ.get('DEEPGRAM_STT_MODEL', 'nova-3')}"
    proc.userdata["llm"] = openai.LLM(model=os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-mini"))
    # TTS backend:
    # - elevenlabs: direct ElevenLabs plugin (default; per LiveKit docs uses ELEVEN_API_KEY)
    # - cartesia: direct Cartesia plugin
    # - livekit: LiveKit Inference Gateway
    tts_backend = os.environ.get("TTS_BACKEND", "elevenlabs").strip().lower()
    if tts_backend == "livekit":
        proc.userdata["tts"] = inference.TTS(
            model=os.environ.get("LK_TTS_MODEL", "cartesia/sonic-2"),
            voice=os.environ.get("LK_TTS_VOICE", ""),
        )
        proc.userdata["tts_model_name"] = f"livekit/{os.environ.get('LK_TTS_MODEL', 'cartesia/sonic-2')}"
    elif tts_backend == "elevenlabs":
        if elevenlabs is None:
            proc.userdata["tts"] = cartesia.TTS(
                model=os.environ.get("CARTESIA_TTS_MODEL", "sonic-2"),
                voice=os.environ.get("CARTESIA_VOICE", "a0e99841-438c-4a64-b679-ae501e7d6091"),
                text_pacing=True,
            )
            proc.userdata["tts_model_name"] = f"cartesia/{os.environ.get('CARTESIA_TTS_MODEL', 'sonic-2')}"
        else:
            # Defaults based on LiveKit ElevenLabs plugin docs.
            # Use eleven_turbo_v2_5 for better quality over phone (vs speed-optimized flash)
            voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL").strip()
            model = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5").strip()
            # ElevenLabs plugin expects api_key via arg or ELEVEN_API_KEY env var, but many setups
            # already have ELEVENLABS_API_KEY configured. Support both to avoid crash-loops.
            api_key = (
                os.environ.get("ELEVEN_API_KEY", "").strip()
                or os.environ.get("ELEVENLABS_API_KEY", "").strip()
            )
            if api_key:
                proc.userdata["tts"] = elevenlabs.TTS(voice_id=voice_id, model=model, api_key=api_key)
            else:
                proc.userdata["tts"] = elevenlabs.TTS(voice_id=voice_id, model=model)
            proc.userdata["tts_model_name"] = f"elevenlabs/{model}"
    else:
        proc.userdata["tts"] = cartesia.TTS(
            model=os.environ.get("CARTESIA_TTS_MODEL", "sonic-2"),
            voice=os.environ.get("CARTESIA_VOICE", "a0e99841-438c-4a64-b679-ae501e7d6091"),
            text_pacing=True,
        )
        proc.userdata["tts_model_name"] = f"cartesia/{os.environ.get('CARTESIA_TTS_MODEL', 'sonic-2')}"


server.setup_fnc = prewarm


async def _entrypoint_impl(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Job received for room %s (inbound/outbound dispatch)", ctx.room.name)

    # Connect to the room so we can read metadata and see participants BEFORE session.start().
    # Use SUBSCRIBE_NONE so we don't pre-subscribe to any tracks — session.start() must
    # manage audio subscriptions itself to properly wire the VAD -> STT -> LLM pipeline.
    #
    # Why this matters for web calls:
    #   - SUBSCRIBE_ALL (default) subscribes to the web user's audio BEFORE session.start()
    #     sets up its audio pipeline, so the audio never reaches VAD/STT.
    #   - AUDIO_ONLY has the same problem.
    #   - SUBSCRIBE_NONE connects (metadata available) but defers track subscription to session.
    #
    # Phone calls work regardless because the SIP participant joins AFTER session.start().
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    # Telephony rooms are created by LiveKit SIP/dispatch rules (not by our API),
    # so they usually DO NOT carry the agent config in room metadata.
    # For web rooms, our API embeds prompt/welcome/voice in room metadata.
    call_id_from_internal: str | None = None
    prompt_from_internal: str | None = None
    welcome_from_internal: dict | None = None
    agent_name_from_internal: str | None = None
    voice_from_internal: dict | None = None
    llm_model_from_internal: str | None = None
    kb_folder_ids_from_internal: list[str] = []
    telephony_trunk_to: str | None = None

    prompt_from_room = _extract_prompt_from_room(ctx)
    welcome_from_room = _extract_welcome_from_room(ctx)
    agent_name_from_room = _extract_agent_name_from_room(ctx)
    llm_model_from_room = _extract_llm_model_from_room(ctx)
    kb_folder_ids_from_room = _extract_knowledge_folder_ids_from_room(ctx)
    bg_audio_from_room = _extract_background_audio_from_room(ctx)

    # Only wait for SIP attrs if we don't already have a prompt in room metadata.
    trunk_to, caller_from = (None, None)
    if not prompt_from_room:
        trunk_to, caller_from = await _wait_for_sip_numbers(ctx)
        if trunk_to:
            telephony_trunk_to = str(trunk_to)
            sip_call_sid = _extract_sip_call_sid(ctx)
            resp = await _post_internal_json_with_retry(
                "/api/internal/telephony/inbound/start",
                {
                    "roomName": ctx.room.name,
                    "to": str(trunk_to),
                    "from": str(caller_from or ""),
                    "twilioCallSid": str(sip_call_sid or ""),
                },
            )
            if isinstance(resp, dict) and resp.get("callId"):
                call_id_from_internal = str(resp.get("callId"))
                prompt_from_internal = resp.get("prompt") if isinstance(resp.get("prompt"), str) else None
                welcome_from_internal = resp.get("welcome") if isinstance(resp.get("welcome"), dict) else None
                agent_name_from_internal = (
                    (resp.get("agent") or {}).get("name") if isinstance(resp.get("agent"), dict) else None
                )
                voice_from_internal = resp.get("voice") if isinstance(resp.get("voice"), dict) else None
                llm_model_from_internal = resp.get("llmModel") if isinstance(resp.get("llmModel"), str) else None
                if isinstance(resp.get("knowledgeFolderIds"), list):
                    kb_folder_ids_from_internal = [
                        str(x).strip()
                        for x in resp.get("knowledgeFolderIds")
                        if isinstance(x, str) and str(x).strip()
                    ][:50]
                max_call_seconds_from_internal = (
                    int(resp.get("maxCallSeconds"))
                    if isinstance(resp.get("maxCallSeconds"), (int, float)) and float(resp.get("maxCallSeconds")) >= 0
                    else None
                )
                logger.info(
                    f"Telephony call linked: callId={call_id_from_internal} to={trunk_to} from={caller_from} "
                    f"promptChars={len(prompt_from_internal or '')} welcomeMode={str((welcome_from_internal or {}).get('mode') or '')}"
                )
            elif trunk_to:
                logger.warning(
                    f"Telephony internal start failed (no callId). to={trunk_to} from={caller_from} resp={resp}"
                )

    extra_prompt = prompt_from_internal or prompt_from_room
    welcome = welcome_from_internal or welcome_from_room
    agent_speaker = (agent_name_from_internal or agent_name_from_room or "Agent").strip()
    llm_model_used = (llm_model_from_internal or llm_model_from_room or "").strip() or None
    kb_folder_ids = kb_folder_ids_from_internal or kb_folder_ids_from_room

    # Max call duration (seconds): from room metadata (web) or internal telephony response (phone).
    max_call_seconds = 0
    try:
        if getattr(ctx.room, "metadata", None):
            meta2 = json.loads(ctx.room.metadata or "{}")
            mcs = (meta2.get("agent") or {}).get("maxCallSeconds")
            if isinstance(mcs, (int, float)) and float(mcs) >= 0:
                max_call_seconds = int(mcs)
    except Exception:
        pass
    try:
        if "max_call_seconds_from_internal" in locals() and max_call_seconds_from_internal is not None:
            max_call_seconds = int(max_call_seconds_from_internal)
    except Exception:
        pass

    llm_obj = ctx.proc.userdata.get("llm")
    try:
        if llm_model_used:
            llm_obj = openai.LLM(model=llm_model_used)
    except Exception:
        # Never fail the call because of an LLM model config issue.
        llm_obj = ctx.proc.userdata.get("llm")

    # Allow per-agent voice config from:
    # 1) room metadata (web sessions)
    # 2) internal telephony response (phone calls)
    tts_obj = ctx.proc.userdata.get("tts")
    tts_model_used = str(ctx.proc.userdata.get("tts_model_name") or "")
    try:
        voice_cfg = {}
        if getattr(ctx.room, "metadata", None):
            try:
                meta = json.loads(ctx.room.metadata or "{}")
                voice_cfg = (meta.get("agent") or {}).get("voice") or {}
            except Exception:
                voice_cfg = {}

        if not voice_cfg and isinstance(voice_from_internal, dict):
            voice_cfg = voice_from_internal

        provider = str(voice_cfg.get("provider") or "").strip().lower()
        model = str(voice_cfg.get("model") or "").strip()
        voice_id = str(voice_cfg.get("voiceId") or "").strip()

        if provider == "cartesia" and voice_id:
            tts_model_used = f"cartesia/{model or os.environ.get('CARTESIA_TTS_MODEL', 'sonic-2')}"
            tts_obj = cartesia.TTS(
                model=model or os.environ.get("CARTESIA_TTS_MODEL", "sonic-2"),
                voice=voice_id,
                text_pacing=True,
            )
        elif provider == "elevenlabs" and voice_id:
            if elevenlabs is None:
                logger.warning("Voice provider is elevenlabs but plugin is not installed; using default TTS.")
            else:
                # ElevenLabs plugin expects api_key via arg or ELEVEN_API_KEY env var.
                api_key = (
                    os.environ.get("ELEVEN_API_KEY", "").strip()
                    or os.environ.get("ELEVENLABS_API_KEY", "").strip()
                    or None
                )
                chosen_model = model or "eleven_turbo_v2_5"
                if api_key:
                    tts_model_used = f"elevenlabs/{chosen_model}"
                    tts_obj = elevenlabs.TTS(voice_id=voice_id, model=chosen_model, api_key=api_key)
                else:
                    tts_model_used = f"elevenlabs/{chosen_model}"
                    tts_obj = elevenlabs.TTS(voice_id=voice_id, model=chosen_model)
    except Exception:
        # Never fail the call because of a voice config issue.
        pass

    # AgentSession — matches the official LiveKit basic_agent pattern.
    # https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
    session = AgentSession(
        stt=ctx.proc.userdata.get("stt"),
        llm=llm_obj,
        tts=tts_obj,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # Room-level track & participant events
    def _is_audio_publication(pub) -> bool:
        kind = getattr(pub, "kind", None)
        # Support enum, int, and string forms across SDK versions.
        if kind in ("audio", "AUDIO", "Audio", 1):
            return True
        # Enum-like objects may expose a 'name'
        name = getattr(kind, "name", None)
        if isinstance(name, str) and name.lower() == "audio":
            return True
        # Fallback: stringified enum value
        try:
            return str(kind).lower().endswith("audio")
        except Exception:
            return False

    def _participant_id(p) -> str:
        return getattr(p, "identity", None) or getattr(p, "sid", None) or "unknown"

    @ctx.room.on("participant_connected")
    def _participant_connected(participant):
        # If tracks were already published before we connected, subscribe now.
        try:
            for pub in participant.track_publications.values():
                if _is_audio_publication(pub) and not getattr(pub, "subscribed", False):
                    pub.set_subscribed(True)
        except Exception:
            pass

    @ctx.room.on("track_published")
    def _track_published(publication, participant):
        # Ensure audio tracks are subscribed even when auto_subscribe=SUBSCRIBE_NONE.
        try:
            if _is_audio_publication(publication) and not getattr(publication, "subscribed", False):
                publication.set_subscribed(True)
        except Exception:
            pass

    transcript_items: list[dict] = []

    @session.on("user_input_transcribed")
    def on_user_input_transcribed_capture(ev):
        # Capture only final segments to keep transcript clean.
        if not getattr(ev, "is_final", False):
            return
        txt = str(getattr(ev, "transcript", "") or "").strip()
        if not txt:
            return
        transcript_items.append(
            {
                "speaker": "User",
                "role": "user",
                "text": txt,
                "final": True,
            }
        )

    # Capture agent output using the documented event type in this SDK version.
    # `conversation_item_added` emits ChatMessage items for both user + assistant;
    # we only persist assistant messages here.
    _last_agent_text: str | None = None

    @session.on("conversation_item_added")
    def on_conversation_item_added_capture(ev):
        nonlocal _last_agent_text
        item = getattr(ev, "item", None)
        if not item:
            return
        role = getattr(item, "role", None)
        if role != "assistant":
            return

        txt = getattr(item, "text_content", None)
        if not txt:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                parts = [c for c in content if isinstance(c, str)]
                txt = "\n".join(parts) if parts else None

        txt = str(txt or "").strip()
        if not txt:
            return
        if _last_agent_text == txt:
            return
        _last_agent_text = txt

        transcript_items.append(
            {
                "speaker": agent_speaker,
                "role": "agent",
                "text": txt,
                "final": True,
            }
        )

    usage_collector = metrics.UsageCollector()
    llm_ttft_ms_sum = 0.0
    llm_ttft_count = 0
    eou_transcription_ms_sum = 0.0
    eou_transcription_count = 0
    eou_end_ms_sum = 0.0
    eou_end_count = 0

    # Participant count sampling (for participant-min billing).
    participant_counts: list[int] = []

    async def sample_participants():
        # Sample until shutdown; cheap, once per second.
        while True:
            try:
                rem = getattr(ctx.room, "remote_participants", None) or {}
                n_remote = len(rem) if hasattr(rem, "__len__") else 0
            except Exception:
                n_remote = 0
            # Count agent + remote participants
            participant_counts.append(max(1, 1 + int(n_remote)))
            await asyncio.sleep(1.0)

    sampler_task = asyncio.create_task(sample_participants())

    async def _stop_sampler():
        try:
            sampler_task.cancel()
        except Exception:
            pass

    ctx.add_shutdown_callback(_stop_sampler)

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        nonlocal llm_ttft_ms_sum, llm_ttft_count, eou_transcription_ms_sum, eou_transcription_count, eou_end_ms_sum, eou_end_count

        if isinstance(ev.metrics, metrics.LLMMetrics):
            if ev.metrics.ttft >= 0:
                llm_ttft_ms_sum += float(ev.metrics.ttft) * 1000.0
                llm_ttft_count += 1
        elif isinstance(ev.metrics, metrics.EOUMetrics):
            eou_transcription_ms_sum += float(ev.metrics.transcription_delay) * 1000.0
            eou_transcription_count += 1
            eou_end_ms_sum += float(ev.metrics.end_of_utterance_delay) * 1000.0
            eou_end_count += 1

    # Telephony setup + voice selection happens before AgentSession init (above).

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

        call_id = call_id_from_internal or _extract_call_id_from_room(ctx)
        if not call_id:
            return

        llm_ttft_ms_avg = (llm_ttft_ms_sum / llm_ttft_count) if llm_ttft_count else None
        eou_transcription_ms_avg = (eou_transcription_ms_sum / eou_transcription_count) if eou_transcription_count else None
        eou_end_ms_avg = (eou_end_ms_sum / eou_end_count) if eou_end_count else None
        # A simple "turn latency" approximation: (transcription delay + LLM TTFT)
        agent_turn_latency_ms_avg = None
        if eou_transcription_ms_avg is not None and llm_ttft_ms_avg is not None:
            agent_turn_latency_ms_avg = eou_transcription_ms_avg + llm_ttft_ms_avg

        participants_count_avg: float | None = None
        if participant_counts:
            try:
                participants_count_avg = float(sum(participant_counts)) / float(len(participant_counts))
            except Exception:
                participants_count_avg = None

        source = "telephony" if call_id_from_internal else "web"

        normalized_payload: dict = {"source": source}
        if participants_count_avg is not None and participants_count_avg > 0:
            normalized_payload["participantsCountAvg"] = participants_count_avg

        telephony_payload: dict = {
            "trunkNumber": str(telephony_trunk_to or ""),
            "callerNumber": str(caller_from or ""),
        }
        sid = str(_extract_sip_call_sid(ctx) or "").strip()
        if sid:
            telephony_payload["twilioCallSid"] = sid

        payload = {
            "usage": asdict(summary),
            "latency": {
                "llm_ttft_ms_avg": llm_ttft_ms_avg,
                "eou_transcription_ms_avg": eou_transcription_ms_avg,
                "eou_end_ms_avg": eou_end_ms_avg,
                "agent_turn_latency_ms_avg": agent_turn_latency_ms_avg,
            },
            "normalized": normalized_payload,
            "telephony": telephony_payload,
        }
        payload["models"] = {}
        if llm_model_used:
            payload["models"]["llm"] = llm_model_used
        stt_model_used = str(ctx.proc.userdata.get("stt_model_name") or "").strip()
        if stt_model_used:
            payload["models"]["stt"] = stt_model_used
        if tts_model_used:
            payload["models"]["tts"] = tts_model_used
        await _post_call_metrics_with_retry(call_id, payload)

    ctx.add_shutdown_callback(log_usage)

    # extra_prompt / welcome / agent_speaker are computed before session init (above).

    _finalized = False
    _finalize_lock = asyncio.Lock()

    async def finalize_call():
        nonlocal _finalized
        async with _finalize_lock:
            if _finalized:
                return
            call_id = call_id_from_internal or _extract_call_id_from_room(ctx)
            if not call_id:
                return
            _finalized = True
            await _post_internal_json_with_retry(
                f"/api/internal/calls/{call_id}/end",
                {"outcome": "completed", "transcript": transcript_items},
            )

    ctx.add_shutdown_callback(finalize_call)

    # Hard stop: if a call exceeds the agent-configured max duration, end it.
    # This prevents "in_progress" calls from hanging forever.
    async def max_duration_watcher():
        try:
            if max_call_seconds and max_call_seconds > 0:
                await asyncio.sleep(float(max_call_seconds))
                if _finalized:
                    return
                try:
                    await finalize_call()
                finally:
                    ctx.shutdown("max_call_duration")
        except asyncio.CancelledError:
            pass
        except Exception:
            # Never crash the job because of the watchdog.
            pass

    if max_call_seconds and max_call_seconds > 0:
        asyncio.create_task(max_duration_watcher())

    async def watch_sip_hangup():
        if not telephony_trunk_to:
            return
        # Wait until we actually see SIP attrs at least once (otherwise we may be too early).
        seen = False
        consecutive_missing = 0
        while True:
            if _finalized:
                return
            t, f = _extract_sip_numbers(ctx)
            if t or f:
                seen = True
                consecutive_missing = 0
            else:
                if seen:
                    consecutive_missing += 1
                    # Require two consecutive misses to avoid transient reads.
                    if consecutive_missing >= 2:
                        try:
                            await finalize_call()
                        finally:
                            # End the job now; don't wait for room empty timeout.
                            ctx.shutdown("sip hangup")
                        return
            await asyncio.sleep(0.4)

    # Start hangup watcher only for telephony calls.
    asyncio.create_task(watch_sip_hangup(), name="watch_sip_hangup")
    
    # session.start() follows the official LiveKit basic_agent pattern:
    # https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
    #
    # Noise cancellation: OFF by default (same as official docs).
    # For telephony only — BVC() on WebRTC/browser audio filters out user speech.
    is_telephony = telephony_trunk_to is not None

    await session.start(
        agent=MyAgent(
            extra_prompt=extra_prompt,
            welcome=welcome,
            call_id=(call_id_from_internal or _extract_call_id_from_room(ctx)),
            kb_folder_ids=kb_folder_ids,
            speaker_name=agent_speaker,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # BVCTelephony noise cancellation — only for telephony (phone lines).
                # NEVER enable for web calls; it filters out browser microphone audio.
                noise_cancellation=noise_cancellation.BVCTelephony() if is_telephony else None,
            ),
            # Sync agent transcription to the room so the web dashboard can display it.
            text_output=room_io.TextOutputOptions(sync_transcription=True),
        ),
    )

    # ── Background audio ──────────────────────────────────────────────
    # Works on both web calls and phone calls. The preset is stored in
    # the agent config and passed via room metadata (agent.backgroundAudio).yes
    bg_audio_cfg = bg_audio_from_room
    if not bg_audio_cfg:
        # For telephony calls, the internal API may pass it back inside voice_from_internal.
        if isinstance(voice_from_internal, dict):
            bg_audio_cfg = voice_from_internal.get("backgroundAudio")
    bg_player = _build_background_audio_player(bg_audio_cfg)
    if bg_player:
        try:
            await bg_player.start(room=ctx.room, agent_session=session)
            logger.info(f"Background audio started: preset={bg_audio_cfg.get('preset')}")
        except Exception as e:
            logger.warning(f"Failed to start background audio: {e}")


if LIVEKIT_AGENT_NAME:
    @server.rtc_session(agent_name=LIVEKIT_AGENT_NAME)
    async def entrypoint(ctx: JobContext):
        await _entrypoint_impl(ctx)
else:
    # When no agent name is provided, rely on LiveKit Cloud dispatch rules / default behavior.
    @server.rtc_session()
    async def entrypoint(ctx: JobContext):
        await _entrypoint_impl(ctx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(f"Starting agent worker (agent_name='{LIVEKIT_AGENT_NAME or ''}')")
    cli.run_app(server)


