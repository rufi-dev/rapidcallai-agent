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
    AutoSubscribe,
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


def _post_call_metrics(call_id: str, payload: dict) -> None:
    base = os.environ.get("SERVER_BASE_URL", "").strip().rstrip("/")
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
        with urlrequest.urlopen(req, timeout=5) as resp:
            _ = resp.read()
    except (HTTPError, URLError) as e:
        logger.warning(f"Failed to post call metrics: {e}")


def _post_internal_json(path: str, payload: dict) -> dict | None:
    base = os.environ.get("SERVER_BASE_URL", "").strip().rstrip("/")
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
        with urlrequest.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except (HTTPError, URLError) as e:
        logger.warning(f"Internal API call failed {path}: {e}")
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
    def __init__(self, *, extra_prompt: str | None = None, welcome: dict | None = None) -> None:
        style = (
            "You interact with users via voice. Keep responses concise and to the point. "
            "Do not use emojis, asterisks, markdown, or other special characters. "
            "Speak English unless the user requests otherwise."
        )

        # IMPORTANT: Do not hardcode a name/persona here; the per-agent prompt should control it.
        if extra_prompt and extra_prompt.strip():
            instructions = (
                "CUSTOM AGENT PROMPT (highest priority):\n"
                f"{extra_prompt.strip()}\n\n"
                "ADDITIONAL STYLE CONSTRAINTS:\n"
                f"{style}"
            )
        else:
            instructions = (
                "You are a helpful voice assistant.\n\n"
                "ADDITIONAL STYLE CONSTRAINTS:\n"
                f"{style}"
            )

        super().__init__(instructions=instructions)
        self._welcome = welcome or {}

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
    else:
        proc.userdata["stt"] = deepgram.STT(
            model=os.environ.get("DEEPGRAM_STT_MODEL", "nova-3"),
            language=os.environ.get("DEEPGRAM_LANGUAGE", "en-US"),
            interim_results=True,
            endpointing_ms=int(float(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "25"))),
            no_delay=True,
            punctuate=True,
            filler_words=True,
        )
    proc.userdata["llm"] = openai.LLM(model=os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-mini"))
    # TTS backend:
    # - livekit: LiveKit Inference Gateway
    # - cartesia: direct Cartesia plugin (recommended when inference is rate-limited)
    tts_backend = os.environ.get("TTS_BACKEND", "cartesia").strip().lower()
    if tts_backend == "livekit":
        proc.userdata["tts"] = inference.TTS(
            model=os.environ.get("LK_TTS_MODEL", "cartesia/sonic-2"),
            voice=os.environ.get("LK_TTS_VOICE", ""),
        )
    else:
        proc.userdata["tts"] = cartesia.TTS(
            model=os.environ.get("CARTESIA_TTS_MODEL", "sonic-2"),
            voice=os.environ.get("CARTESIA_VOICE", "a0e99841-438c-4a64-b679-ae501e7d6091"),
            text_pacing=True,
        )


server.setup_fnc = prewarm


async def _entrypoint_impl(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Ensure we are connected and have up-to-date room metadata / participant linkage.
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        # Use provider plugins directly (avoids LiveKit hosted inference quota).
        stt=ctx.proc.userdata.get("stt"),
        llm=ctx.proc.userdata.get("llm"),
        tts=ctx.proc.userdata.get("tts"),
        vad=ctx.proc.userdata["vad"],
        # Reduce audio "cutting" caused by aggressive interruption detection in noisy rooms.
        # Console mode feels smoother because it doesn't have RTC echo/noise artifacts.
        allow_interruptions=os.environ.get("LK_ALLOW_INTERRUPTIONS", "true").lower() == "true",
        min_interruption_duration=_env_float("LK_MIN_INTERRUPTION_DURATION", 0.9),
        min_interruption_words=int(float(os.environ.get("LK_MIN_INTERRUPTION_WORDS", "2"))),
        # Reduce "wait after you stop talking" time before the agent answers.
        turn_detection="vad",
        min_endpointing_delay=_env_float("LK_MIN_ENDPOINTING_DELAY", 0.15),
        max_endpointing_delay=_env_float("LK_MAX_ENDPOINTING_DELAY", 0.8),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=_env_float("LK_FALSE_INTERRUPTION_TIMEOUT", 1.0),
    )

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        logger.info(f"User state: {ev.old_state} -> {ev.new_state}")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        logger.info(f"Agent state: {ev.old_state} -> {ev.new_state}")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        logger.info(f"User transcript ({'final' if ev.is_final else 'partial'}): {ev.transcript}")

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

    # Best-effort capture of agent output (event name may vary by SDK version).
    @session.on("agent_output_transcribed")
    def on_agent_output_transcribed_capture(ev):
        txt = str(getattr(ev, "transcript", "") or getattr(ev, "text", "") or "").strip()
        if not txt:
            return
        transcript_items.append(
            {
                "speaker": "Agent",
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

    # If this is a telephony call, create a call record and fetch the agent prompt via internal API.
    call_id_from_internal: str | None = None
    prompt_from_internal: str | None = None
    welcome_from_internal: dict | None = None

    trunk_to, caller_from = await _wait_for_sip_numbers(ctx)
    if trunk_to:
        resp = await asyncio.to_thread(
            _post_internal_json,
            "/api/internal/telephony/inbound/start",
            {"roomName": ctx.room.name, "to": str(trunk_to), "from": str(caller_from or "")},
        )
        if isinstance(resp, dict) and resp.get("callId"):
            call_id_from_internal = str(resp.get("callId"))
            prompt_from_internal = resp.get("prompt") if isinstance(resp.get("prompt"), str) else None
            welcome_from_internal = resp.get("welcome") if isinstance(resp.get("welcome"), dict) else None
            logger.info(
                f"Telephony call linked: callId={call_id_from_internal} to={trunk_to} from={caller_from} "
                f"promptChars={len(prompt_from_internal or '')} welcomeMode={str((welcome_from_internal or {}).get('mode') or '')}"
            )
        else:
            logger.warning(f"Telephony internal start failed (no callId). to={trunk_to} from={caller_from} resp={resp}")

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

        payload = {
            "usage": asdict(summary),
            "latency": {
                "llm_ttft_ms_avg": llm_ttft_ms_avg,
                "eou_transcription_ms_avg": eou_transcription_ms_avg,
                "eou_end_ms_avg": eou_end_ms_avg,
                "agent_turn_latency_ms_avg": agent_turn_latency_ms_avg,
            },
        }
        await asyncio.to_thread(_post_call_metrics, call_id, payload)

    ctx.add_shutdown_callback(log_usage)

    extra_prompt = prompt_from_internal or _extract_prompt_from_room(ctx)
    welcome = welcome_from_internal or _extract_welcome_from_room(ctx)

    async def finalize_call():
        call_id = call_id_from_internal or _extract_call_id_from_room(ctx)
        if not call_id:
            return
        await asyncio.to_thread(
            _post_internal_json,
            f"/api/internal/calls/{call_id}/end",
            {"outcome": "completed", "transcript": transcript_items},
        )

    ctx.add_shutdown_callback(finalize_call)
    await session.start(
        agent=MyAgent(extra_prompt=extra_prompt, welcome=welcome),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # Smaller frames reduce end-to-end latency for both STT and agent response timing.
            audio_input=room_io.AudioInputOptions(
                frame_size_ms=int(float(os.environ.get("LK_AUDIO_FRAME_MS", "20"))),
                pre_connect_audio=True,
                pre_connect_audio_timeout=_env_float("LK_PRECONNECT_AUDIO_TIMEOUT", 2.0),
            ),
            # Publish transcriptions to the room for clients to render.
            # Keep agent output text synced to TTS audio so the UI doesn't "dump" the full response at once.
            # (User STT interim results are controlled by the STT backend settings, not this flag.)
            text_output=room_io.TextOutputOptions(sync_transcription=True),
        ),
    )


@server.rtc_session(agent_name=LIVEKIT_AGENT_NAME)
async def entrypoint(ctx: JobContext):
    await _entrypoint_impl(ctx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(f"Starting agent worker (agent_name='{LIVEKIT_AGENT_NAME or ''}')")
    cli.run_app(server)


