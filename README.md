# yt-stock-analysis

A production-style pipeline that turns popular stock-market YouTube videos into
structured, fact-checked research.

Two modes:

| Mode | Command | What it does |
|---|---|---|
| **Daily digest** | `ytstock run` | Finds the day's most-viewed stock-market videos, pulls transcripts, extracts a structured analysis per video with Claude, and writes a cross-video digest (consensus, ticker table, predictions to track). |
| **Trading brief** | `ytstock brief <urls…>` | Takes videos *you* pick, extracts insights, **fact-checks the claims with web search**, and writes a trading brief: macro, market, single names and derivatives, hidden second-order ideas, ranked trade ideas with entry/invalidation, and a dated watch list for the next few days. |

Everything is idempotent and resumable: each stage is keyed by video id, failures
are recorded per video, and re-running a day only does the missing work.

## No API keys required (but supported)

| Need | Free / keyless path (default) | Keyed path |
|---|---|---|
| Find videos | `yt-dlp` scraping of YouTube search + channel pages | YouTube Data API v3 (`YOUTUBE_API_KEY`; free 10k units/day) |
| Video metadata | `yt-dlp`, then oEmbed | YouTube Data API |
| Transcripts | `youtube-transcript-api` (captions) | same, plus optional Whisper fallback |
| Claude | **`claude -p` on your Claude Code subscription** (`LLM_BACKEND=claude-cli`) | Anthropic API (`ANTHROPIC_API_KEY`) |

`auto` (default) picks the keyed path when a key is present, otherwise the keyless one.

When to use a key anyway:
- **YouTube Data API** gives exact publish times and stats in one call and is more
  robust than scraping; yt-dlp discovery needs one extra request per candidate
  (~2–3 s each, run in parallel) and can break when YouTube changes its pages.
- **Anthropic API** gives structured outputs, prompt caching, effort control,
  server-side refusal fallbacks, exact cost accounting, and runs unattended in
  GitHub Actions. The `claude -p` backend is great for local use but is bound by
  your plan's rate limits, adds Claude Code's own system-prompt overhead to every
  call, and cannot run in CI without your login.

## Quick start

```bash
git clone https://github.com/shuaidop/yt-stock-analysis.git && cd yt-stock-analysis
uv sync                    # or: pip install -e .
cp .env.example .env       # optional; everything has a default

# Brief mode on videos you choose (keyless + Claude Code subscription)
uv run ytstock brief "https://www.youtube.com/watch?v=VIDEO1" "https://youtu.be/VIDEO2"

# Daily digest for today (keyless discovery via yt-dlp)
uv run ytstock run --top 10

# Inspect
uv run ytstock status --date 2026-09-02
cat reports/latest-brief.md
```

The first Claude call goes through `claude -p` if you are logged into Claude Code
and `ANTHROPIC_API_KEY` is unset. Set `LLM_BACKEND=api` / `claude-cli` to force one.

### Transcripts from cloud IPs

YouTube blocks caption requests from most datacenter IPs. Locally this is rarely
an issue; in GitHub Actions or on a VPS set `YT_PROXY_URL` (any HTTP proxy) or the
`WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD` pair.

### Videos without captions (Whisper fallback)

Many creator videos, especially Chinese-language ones, have no captions at all.
Set `WHISPER_FALLBACK=true` and the pipeline downloads the audio with yt-dlp and
transcribes it locally (needs `ffmpeg`). Two backends:

| Backend | Install | Speed / quality (M2 Max, 23-min Chinese video) |
|---|---|---|
| `mlx` (Apple Silicon GPU, default when importable) | `uv sync --extra mlx` | `large-v3-turbo`: 66 s, finance terms correct |
| `faster` (CPU, any platform) | `uv sync --extra whisper` | `small`: 4.6 min, frequent homophone errors on jargon |

Captions in other languages are kept in the original language (Claude reads them
natively); `REPORT_LANGUAGE` controls the language of the written output.

## Pipeline

```
 discover ──► transcribe ──► analyze ──► report            (daily digest)
 ingest   ──► transcribe ──► analyze ──► factcheck ──► brief (trading brief)
```

| Stage | Module | Notes |
|---|---|---|
| discover | `discovery.py`, `youtube.py`, `ytdlp_discovery.py` | Search queries + channel watchlist from `configs/sources.yaml`; filtered by window, duration, views, keywords; ranked by views. |
| ingest | `youtube.py` | Parses URLs / ids; metadata via Data API, yt-dlp, or oEmbed. |
| transcribe | `transcripts.py` | Manual captions → auto captions → translated captions → (optional) Whisper. Transient blocks retried; permanent failures stop after 3 attempts. |
| analyze | `analysis.py` | One structured-output call per video (`VideoAnalysis`: outlook, themes, tickers with sentiment/timeframe, price levels, claims with resolve-by dates, recommendations, risk flags, signal quality). Ticker symbols normalised (SPX→SPY etc.). |
| factcheck | `analysis.py` | Per video, Claude + web search verifies claims: supported / partial / contradicted / unverifiable / not-yet-resolved, with sources, corrections, and new context since recording. |
| report | `report.py` | Digest markdown + JSON; a second Claude call writes the cross-video synthesis. |
| brief | `report.py` | Trading brief markdown + JSON (`TradingBrief` schema). |

Outputs land in `reports/` (`YYYY-MM-DD.md/.json`, `brief-YYYY-MM-DD.md/.json`,
`latest.md`, `latest-brief.md`). All raw results, token usage, and cost per call
are in the SQLite database (`data/ytstock.db`; set `DATABASE_URL` for Postgres).

## CLI

```
ytstock run        [--date D] [--top N] [--skip-discovery]     full daily pipeline
ytstock brief URL… [--date D] [--no-factcheck] [--include-daily]
ytstock discover | transcribe | analyze [--force] | factcheck [--force] | report [--no-synthesis]
ytstock status [--date D]        per-stage counts, cost, recent stage runs
ytstock show VIDEO_ID            stored analysis JSON for one video
ytstock init-db | sources
```

`--date` defaults to today in `MARKET_TIMEZONE`, or yesterday before 06:00 local
(so an overnight cron picks up the previous session's recap videos).

## Configuration

All settings are environment variables (or `.env`); see `.env.example` for the
full list. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `auto` | `api`, `claude-cli`, or auto-detect |
| `CLAUDE_MODEL` | `claude-opus-5` | Model for all Claude calls |
| `CLAUDE_VIDEO_EFFORT` / `_SYNTHESIS_EFFORT` / `_FACTCHECK_EFFORT` / `_BRIEF_EFFORT` | medium / high / high / xhigh | Adaptive-thinking effort per stage (API backend) |
| `CLAUDE_FALLBACKS` | `true` | Server-side refusal fallbacks (API backend, beta) |
| `YOUTUBE_BACKEND` | `auto` | `api`, `ytdlp`, or auto-detect |
| `TOP_N_VIDEOS` | 10 | Videos per day |
| `MIN_VIEW_COUNT` / `MIN_DURATION_SECONDS` | 1000 / 180 | Discovery filters |
| `ANALYSIS_CONCURRENCY` | 3 | Parallel Claude calls |

Sources (search queries, channel watchlist, keyword filters) live in
`configs/sources.yaml`.

### Cost

With the API backend and Opus 5, a 15-minute video costs roughly $0.10–0.20 to
analyse and $0.30–0.80 to fact-check (web search); the brief is $0.50–1.50. A
10-video daily digest lands around $2. Per-call token usage and estimated cost
are stored on every row and totalled in `ytstock status`. Under `claude-cli`
the same figures are list-price equivalents, not what you are billed.

## Scheduling

`.github/workflows/daily.yml` runs `ytstock run` Tue–Sat 03:30 UTC (23:30 ET
Mon–Fri), keeps the SQLite database in the Actions cache, commits the markdown
report to `reports/`, and uploads it as an artifact. Add repository secrets
`YOUTUBE_API_KEY`, `ANTHROPIC_API_KEY`, and a transcript proxy (`YT_PROXY_URL`
or the Webshare pair); the runner's IP will otherwise be blocked for captions.

For a non-GitHub deployment, `Dockerfile` builds a self-contained image
(`docker run -v ytstock:/data yt-stock-analysis run`).

## Development

```bash
uv sync --all-groups
make check          # ruff + pytest (coverage)
make typecheck      # mypy
```

Tests mock every network boundary (YouTube HTTP, transcript API, Claude SDK,
`claude -p` subprocess) and run in under a second. The `PROMPT_VERSION`
constants in `analysis.py` are stored with every result; bump one when you
change a prompt or schema and the pipeline re-analyses only what changed.

## Disclaimer

Output attributes views to the creators and to the sources found while
fact-checking. It is research tooling, not investment advice.
