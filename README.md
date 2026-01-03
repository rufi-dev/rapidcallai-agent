## Python LiveKit Agent

This folder contains the LiveKit Agents (Python) code.

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

Notes:
- The agent reads the selected “agent prompt” from the LiveKit room metadata (set by the Node server).


