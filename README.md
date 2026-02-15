## Python LiveKit Agent

This folder contains the LiveKit Agents (Python) code.

It is deployed to **LiveKit Cloud** and runs your voice agent logic. The API creates rooms and embeds:

- the selected agent prompt
- welcome configuration
- callId

in LiveKit room metadata. This agent reads that metadata on join.

**Custom background audio:** Place `Office1.mp3` and `Office2.mp3` in `audio/` to use the **Office 1 (custom)** and **Office 2 (custom)** presets in the dashboard. See `audio/README.md`.

### Local run (console)

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
py -3.13 -m pip install -r requirements.txt
python agent.py console
```

### Cloud deploy (uses your existing LiveKit Cloud agent)

From `python-agent/` (recommended):

```bash
cd python-agent
lk agent deploy --secrets-file .env .
```
python agent.py start

Notes:
- The agent reads the selected “agent prompt” from the LiveKit room metadata (set by the Node server).
- For metrics posting to work from LiveKit Cloud, you MUST set:
  - `SERVER_BASE_URL=https://api.rapidcall.ai` (your public API URL)

## Updating the cloud agent (when code changes)

1) Commit & push to GitHub repo `rufi-dev/rapidcallai-agent`.
2) Redeploy to LiveKit Cloud:

```bash
cd python-agent
lk agent deploy --secrets-file .env .
```

## Troubleshooting

### Nothing in logs when I call the number

If you call your inbound number and **no new lines** appear in the agent terminal (no "received job request", no "Job received for room ..."):

1. **Only one worker with that name**  
   If you have **another** process or **cloud-deployed agent** that also registers as `VoiceAgent` on the **same** LiveKit project, LiveKit may dispatch the call to that worker instead of this one. To see jobs in this terminal, stop the other agent, or use a different `LIVEKIT_AGENT_NAME` for local testing and a dispatch rule that targets it.

2. **Agent still connected**  
   Confirm you see `registered worker` with `agent_name: "VoiceAgent"` in the logs. If the agent disconnected or crashed, restart it and try again.

3. **Same LiveKit project**  
   This process must use the **same** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` as the LiveKit project where your SIP trunk and dispatch rule are configured. Otherwise the job never reaches this worker.

4. **Room created?**  
   In **LiveKit Cloud** → your project → **Rooms** (or Telephony), place a test call and check whether a **room** is created (e.g. `call-+...`). If no room appears, the call is not reaching LiveKit (check Twilio origination). If a room appears but no agent joins, the dispatch rule or agent name may not match.

### "failed to synthesize speech" / "no audio frames were pushed" (agent has no voice)

The agent hears you (STT works) and the LLM replies, but **no speech** is played. The log shows `livekit.plugins.elevenlabs.tts.TTS` and `APIError: no audio frames were pushed for text: ...`.

**Cause:** ElevenLabs TTS is not returning audio. Common reasons: missing/invalid `ELEVEN_API_KEY` or `ELEVENLABS_API_KEY`, quota exceeded, or invalid voice ID.

**Quick fix — use Cartesia instead:** In `python-agent/.env` set `TTS_BACKEND=cartesia` and `CARTESIA_API_KEY=<your key>`, then restart the agent. If no ElevenLabs key is set, the agent already falls back to Cartesia automatically.

### Agent can’t post metrics

If the agent runs in LiveKit Cloud, it cannot reach `http://localhost:8787`.
Use a public URL:

- `SERVER_BASE_URL=https://api.rapidcall.ai`


