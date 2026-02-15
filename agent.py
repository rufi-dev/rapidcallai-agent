"""
Minimal LiveKit voice agent. Follows official examples only.
- basic_agent: https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
- mcp-agent: https://github.com/livekit/agents/blob/main/examples/voice_agents/mcp/mcp-agent.py
- Session/docs: https://docs.livekit.io/agents/build/sessions/
"""
import json
import logging
import os

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
    inference,
    metrics,
    room_io,
)
from livekit.agents.beta.tools.end_call import EndCallTool
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit import rtc
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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


class VoiceAgent(Agent):
    """Single agent with instructions and end_call tool (per LiveKit docs)."""

    def __init__(self, instructions: str, speak_first: bool = True) -> None:
        super().__init__(
            instructions=instructions,
            tools=[EndCallTool()],
        )
        self._speak_first = speak_first

    async def on_enter(self):
        if not self._speak_first:
            return
        # Give client time to calibrate AEC (basic_agent.py)
        self.session.generate_reply(allow_interruptions=False)

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

    instructions = _instructions_from_room(ctx) or (
        "You are a helpful voice assistant. Keep responses concise. "
        "Do not use emojis or markdown. Speak naturally for TTS."
    )
    speak_first = _welcome_mode_from_room(ctx) != "user"

    # Session options from basic_agent + docs (preemptive_generation, resume_false_interruption).
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
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

    await session.start(
        agent=VoiceAgent(instructions=instructions, speak_first=speak_first),
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
