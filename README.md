# Python LiveKit Agent (minimal)

Minimal voice agent aligned with official LiveKit examples only.

- **basic_agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py  
- **mcp-agent:** https://github.com/livekit/agents/blob/main/examples/voice_agents/mcp/mcp-agent.py  
- **Session docs:** https://docs.livekit.io/agents/build/sessions/

## What this agent does

- Reads **prompt** from room metadata (`agent.prompt`). Your API sets this when creating the room.
- Uses **inference** models: Deepgram nova-3 (STT), OpenAI gpt-4.1-mini (LLM), Cartesia sonic-3 (TTS).
- **Voice:** Reads `agent.voice` from room metadata (provider, model, voiceId). The dashboard Voice tab selection is sent when you start Talk and is used for TTS (Cartesia or ElevenLabs).
- **Tools:** Reads `agent.enabledTools` from room metadata. Use the dashboard **Tools** tab to enable/disable **end_call** and **lookup_weather**. The agent only loads tools that are enabled.
- **End call:** When enabled in Tools, the agent can hang up when the user says goodbye. Add to your prompt: *"When the user says goodbye, say a brief goodbye and use the end_call tool."*
- **MCP (optional):** Set `MCP_SERVER_URL=http://localhost:8000/sse` and run `mcp_server.py`. Install with `pip install 'livekit-agents[mcp]'`.
- **Plugins:** See `requirements.txt` for optional plugins (anthropic, assemblyai, google, groq, nvidia, etc.). Uncomment as needed.

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
