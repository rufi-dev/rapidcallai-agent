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
from livekit.plugins import silero

load_dotenv()
logger = logging.getLogger("basic-agent")


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
    url = f"{base}/api/calls/{call_id}/metrics"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            _ = resp.read()
    except (HTTPError, URLError) as e:
        logger.warning(f"Failed to post call metrics: {e}")


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


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Ensure we are connected and have up-to-date room metadata / participant linkage.
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
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

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

        call_id = _extract_call_id_from_room(ctx)
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

    extra_prompt = _extract_prompt_from_room(ctx)
    welcome = _extract_welcome_from_room(ctx)
    await session.start(
        agent=MyAgent(extra_prompt=extra_prompt, welcome=welcome),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            text_output=True,  # publish transcriptions to the room for clients to render
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cli.run_app(server)


