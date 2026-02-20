# Python LiveKit Agent (minimal)

Minimal voice agent aligned with official LiveKit examples only.

- **basic_agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py  
- **mcp-agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/mcp/mcp-agent.py  
- **nvidia_test:** https://github.com/livekit/agents/blob/main/examples/voice_agents/nvidia_test.py  
- **flush_llm_node:** https://github.com/livekit/agents/blob/main/examples/voice_agents/flush_llm_node.py  
- **Session docs:** https://docs.livekit.io/agents/build/sessions/

## What this agent does

- Reads **prompt** from room metadata (`agent.prompt`). Your API sets this when creating the room.
- **STT/TTS:** Deepgram (default), or NVIDIA when `agent.voice.provider` is `"nvidia"`. TTS: Cartesia, ElevenLabs, or NVIDIA from `agent.voice`. Set `NVIDIA_API_KEY` for NVIDIA.
- **Voice:** Reads `agent.voice` from room metadata (provider, model, voiceId). Use provider `"nvidia"` for NVIDIA STT+TTS (nvidia_test pattern).
- **Tools:** Reads `agent.enabledTools` from room metadata. Use the dashboard **Tools** tab to enable/disable **end_call** and **lookup_weather**. When a tool is invoked, the agent says a quick “One moment while I look that up.” and flushes to TTS (flush_llm_node pattern).
- **End call:** When enabled in Tools, the agent can hang up when the user says goodbye. Add to your prompt: *"When the user says goodbye, say a brief goodbye and use the end_call tool."*
- **Call options (inactivity & max duration):** Reads `agent.callOptions` from room metadata. Optional: `maxCallDurationMinutes`, `inactivityCheckSeconds`, `inactivityPrompt` (e.g. "Are you still there?"), `endCallAfterInactivitySeconds`. If the user is silent for `inactivityCheckSeconds`, the agent says `inactivityPrompt`; if still silent for `endCallAfterInactivitySeconds` after that, the call ends. If `maxCallDurationMinutes` is set, the call ends when the limit is reached.
- **MCP (optional):** Set `MCP_SERVER_URL=http://localhost:8000/sse` and run `mcp_server.py`. Install with `pip install 'livekit-agents[mcp]'`.
- **Plugins:** See `requirements.txt` for optional plugins (anthropic, assemblyai, google, groq, nvidia, etc.). NVIDIA is included for STT/TTS.

## Environment (phone / server API)

For phone calls the agent calls your server for config (inbound/start or outbound/start). Set **either** `SERVER_BASE_URL` **or** `PUBLIC_API_BASE_URL` to your API base (e.g. `https://api.rapidcall.ai`) and `AGENT_SHARED_SECRET` to the same value as on the server. The server does not need `SERVER_BASE_URL`; it only needs `AGENT_SHARED_SECRET` and optionally `PUBLIC_API_BASE_URL` for links.

## Local run

```bash
cd python-agent
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python agent.py console
```

## MCP server (optional)

Run in a separate terminal:

```bash
pip install mcp
python mcp_server.py
```

Then set `MCP_SERVER_URL=http://localhost:8000/sse` in `.env` so the agent connects to the MCP server.

## Cloud deploy

```bash
cd python-agent
lk agent deploy --secrets-file .env .
```

Set `LIVEKIT_AGENT_NAME` in `.env` if you use agent dispatch.

## Troubleshooting

- **No speech:** Set `CARTESIA_API_KEY` (agent uses Cartesia TTS by default).
- **Agent not hearing user (web):** The agent uses `SUBSCRIBE_NONE` and subscribes to remote audio after connect; if you change the entrypoint, keep that behavior.
- **Robotic sound on your site:** Use a single `RoomAudioRenderer` and the “Start audio” unlock (browser autoplay). See client Talk UI.

### Why calendar/slots don't work on phone calls (logs)

When the user asks for available dates/slots on a **phone** call and gets "I can't check the calendar", the agent didn't receive tool config. Use these log searches:

**Agent logs** — search for:
- `[check_availability_cal] not_configured on phone call` — shows `toolConfigKeysReceived=` and `configFetchSkipReason=` (why config was missing).
- `Inbound config: got agent config for` — inbound/start or outbound/start succeeded; if you never see this on phone, config is failing.
- For **outbound** rooms (`out-*`), the agent calls `outbound/start` by room name (no dialed number needed).
- `Inbound config: could not get 'to' from` — agent couldn't get dialed number; see `(tried E.164s: ...)`.
- `Inbound config: SIP participant attributes for room` — shows what LiveKit sent (e.g. sip.trunkPhoneNumber).
- `Inbound config: skip (no SERVER_BASE_URL or AGENT_SHARED_SECRET)` — set both env vars on the agent.

**Server logs** — search for:
- `[internal.telephony.inbound.start] request received` — agent called inbound/start (if missing, agent isn't calling API).
- `[internal.telephony.inbound.start]` with roomName, to, from — request params.
- `[internal.telephony.inbound.start] room metadata updated` — server set room metadata.
- `[internal.telephony.inbound.start] phone number not found` — the "to" sent isn't in your DB.
- For **outbound**: `[internal.telephony.outbound.start] request received` / `returning config for outbound call` — agent got config by room name; `call not found for room` means no call record for that room yet.
