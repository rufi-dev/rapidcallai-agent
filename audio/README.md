# Custom background audio

Place your custom MP3 files here. The agent will use them when the user selects **Office 1** or **Office 2** in the dashboard Background Audio settings.

**Required files (you upload):**

- `Office1.mp3` — used for preset **Office 1 (custom)**
- `Office2.mp3` — used for preset **Office 2 (custom)**

Supported formats: MP3, WAV, AAC, FLAC, OGG, Opus (decoded via PyAV in the agent). MP3 is recommended. The agent depends on the `av` (PyAV) package for decoding; see `requirements.txt`.

**Deployment:**

- **Local run:** Put the files in this `audio/` folder next to `agent.py`. Run the agent from the `python-agent/` directory.
- **Docker / LiveKit Cloud:** Add the files to this folder **before** building the image. The Dockerfile copies the whole project (including `audio/`). Rebuild and redeploy after adding or changing the MP3s.

**No sound in call?**

1. Check agent logs for:
   - `Background audio file not found: path=...` → the file is missing at that path (wrong deploy or path).
   - `Background audio dir: ... (exists=True)` and `Office1.mp3: found` → files are present; if you still hear nothing, look for `Failed to start background audio:` and the exception (e.g. missing `av` for MP3).
2. Ensure the agent preset is **Office 1 (custom)** or **Office 2 (custom)** and you saved the agent in the dashboard.
3. Rebuild the agent image after adding the MP3s so they are inside the image.
