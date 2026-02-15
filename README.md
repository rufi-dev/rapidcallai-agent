# Python LiveKit Agent (minimal)

Minimal voice agent aligned with official LiveKit examples only.

- **basic_agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py  
- **mcp-agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/mcp/mcp-agent.py  
- **Session docs:** https://docs.livekit.io/agents/build/sessions/

## What this agent does

- Reads **prompt** from room metadata (`agent.prompt`). Your API sets this when creating the room.
- Uses **inference** models: Deepgram nova-3 (STT), OpenAI gpt-4.1-mini (LLM), Cartesia sonic-3 (TTS).
- **End call:** Exposes LiveKit’s `EndCallTool` so the LLM can end the call when the user says goodbye.
- **MCP (optional):** Set `MCP_SERVER_URL=http://localhost:8000/sse` and run `mcp_server.py` to add tools (e.g. `get_weather`). Install with `pip install 'livekit-agents[mcp]'`.

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
