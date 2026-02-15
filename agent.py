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
import time
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
    """MCP servers if URL set and livekit-agents[mcp] installed (see mcp-agent.py example)."""
    if not MCP_SERVER_URL:
        return []
    try:
        from livekit.agents import mcp
        return [mcp.MCPServerHTTP(url=MCP_SERVER_URL)]
    except ImportError:
        logger.warning("MCP_SERVER_URL set but livekit-agents[mcp] not installed; pip install 'livekit-agents[mcp]'")
        return []


def _instructions_from_room(ctx: JobContext) -> str | None:
    """Read prompt from room metadata. LiveKit room metadata is set by your server."""
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
    """Read welcome.mode from room metadata: 'user' = user speaks first, else AI speaks first."""
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
    """Read agent.voice from room metadata so dashboard voice selection is used."""
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
    """Read agent.enabledTools from room metadata (dashboard Tools tab). Default: end_call only."""
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
    """Read agent.backchannelEnabled from room metadata (dashboard Speech settings)."""
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


def _positive_float(val, default: float) -> float:
    try:
        v = float(val) if val is not None else default
        return max(0.0, v)
    except (TypeError, ValueError):
        return default


def _build_stt(ctx: JobContext):
    """STT from room metadata: use NVIDIA when voice.provider is nvidia, else Deepgram."""
    voice_cfg = _voice_from_room(ctx)
    provider = str(voice_cfg.get("provider") or "").strip().lower()
    if provider == "nvidia" and nvidia:
        lang = str(voice_cfg.get("languageCode") or "en-US").strip() or "en-US"
        return nvidia.STT(language_code=lang)
    return inference.STT("deepgram/nova-3", language="multi")


def _build_tts(ctx: JobContext):
    """TTS from room metadata voice (provider, model, voiceId) so dashboard voice selection is used."""
    voice_cfg = _voice_from_room(ctx)
    provider = str(voice_cfg.get("provider") or "").strip().lower()
    model = str(voice_cfg.get("model") or "").strip() or "sonic-3"
    voice_id = str(voice_cfg.get("voiceId") or "").strip() or "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

    if provider == "nvidia" and nvidia:
        # NVIDIA TTS: voice is e.g. Magpie-Multilingual.EN-US.Leo or from dashboard
        nvidia_voice = voice_id if voice_id and not voice_id.startswith("96") else "Magpie-Multilingual.EN-US.Leo"
        lang = str(voice_cfg.get("languageCode") or "en-US").strip() or "en-US"
        return nvidia.TTS(voice=nvidia_voice, language_code=lang)
    if provider == "elevenlabs" and elevenlabs and voice_id:
        return elevenlabs.TTS(
            voice_id=voice_id,
            model=model or "eleven_turbo_v2_5",
        )
    return cartesia.TTS(model=model or "sonic-3", voice=voice_id, text_pacing=True)


class VoiceAgent(Agent):
    """Single agent with instructions and tools from dashboard (end_call, lookup_weather).
    Uses flush llm_node: when a tool is invoked without prior text, says a quick filler and flushes to TTS.
    """

    def __init__(
        self,
        instructions: str,
        speak_first: bool = True,
        tools: list | None = None,
    ) -> None:
        tools = tools or [EndCallTool()]
        super().__init__(instructions=instructions, tools=tools)
        self._speak_first = speak_first

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

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather. Provide location; do not ask for lat/long."""
        logger.info("lookup_weather: %s", location)
        return "Sunny, 70°F."


server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm VAD (basic_agent.py)."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


async def _run_session(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Connect with SUBSCRIBE_NONE so session manages audio (avoids agent not hearing user on web).
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)

    base = _instructions_from_room(ctx) or (
        "You are a helpful voice assistant. Keep responses concise. "
        "Do not use emojis or markdown. Speak naturally for TTS."
    )
    # Voice-only: no numbered or bullet lists (causes robotic pauses on web). Use flowing sentences.
    voice_rules = (
        "VOICE OUTPUT: Reply in short, flowing sentences. Do not use numbered lists (1. 2. 3.) "
        "or bullet points; they cause long pauses when spoken. Say the same content in plain prose."
    )
    # End call: document the tool so the LLM (and you in the dashboard prompt) know when to use it.
    end_call_rule = (
        "END CALL: You have an end_call tool. When the user says goodbye, wants to hang up, or "
        "is done with the conversation, call end_call. Say a brief goodbye first, then call the tool."
    )
    backchannel_rule = ""
    if _backchannel_enabled_from_room(ctx):
        backchannel_rule = (
            "\n\nBACKCHANNELING (important): When the user is speaking or has just paused mid-thought "
            "(e.g. ended with \"and\", \"so\", \"but\", or an incomplete sentence), respond with ONLY a very short "
            "listener cue: \"yeah\", \"uh-huh\", \"mm-hmm\", \"right\", \"mhm\", or \"I see\". Do this DURING the "
            "conversation whenever the user is sharing a story or long message—offer one short backchannel, then "
            "let them continue. Keep your full replies for when they ask a direct question or finish a complete thought."
        )
    instructions = f"{base}\n\n{voice_rules}\n\n{end_call_rule}{backchannel_rule}"
    speak_first = _welcome_mode_from_room(ctx) != "user"
    enabled = _enabled_tools_from_room(ctx)
    tool_list = [EndCallTool()] if "end_call" in enabled else []
    # lookup_weather is on VoiceAgent as @function_tool; add agent only if that tool is enabled
    tts_obj = _build_tts(ctx)
    stt_obj = _build_stt(ctx)

    session = AgentSession(
        stt=stt_obj,
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts_obj,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,
        mcp_servers=_mcp_servers(),
    )

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

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # Inactivity and max call duration (from room metadata agent.callOptions)
    call_opts = _call_options_from_room(ctx)
    last_activity = {"t": time.monotonic()}
    session_start = time.monotonic()
    inactivity_prompted_at: float | None = None

    def _on_user_input(ev):
        if getattr(ev, "is_final", True):
            last_activity["t"] = time.monotonic()

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

    await session.start(
        agent=VoiceAgent(instructions=instructions, speak_first=speak_first, tools=tool_list),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
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
