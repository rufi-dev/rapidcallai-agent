# Custom background audio

Place your custom MP3 files here. The agent will use them when the user selects **Office 1** or **Office 2** in the dashboard Background Audio settings.

**Required files (you upload):**

- `Office1.mp3` — used for preset **Office 1 (custom)**
- `Office2.mp3` — used for preset **Office 2 (custom)**

Supported formats: MP3, WAV, AAC, FLAC, OGG, Opus (via LiveKit/FFmpeg). MP3 is recommended.

**Deployment:**

- **Local run:** Put the files in this `audio/` folder next to `agent.py`.
- **Docker:** The Dockerfile copies this folder into the image. Add the files before building, or mount a volume over `/app/audio` with your files.
