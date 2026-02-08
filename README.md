## Python LiveKit Agent

This folder contains the LiveKit Agents (Python) code.

It is deployed to **LiveKit Cloud** and runs your voice agent logic. The API creates rooms and embeds:

- the selected agent prompt
- welcome configuration
- callId

in LiveKit room metadata. This agent reads that metadata on join.

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

### Agent can’t post metrics

If the agent runs in LiveKit Cloud, it cannot reach `http://localhost:8787`.
Use a public URL:

- `SERVER_BASE_URL=https://api.rapidcall.ai`


