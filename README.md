# μ-Bench

Author and maintainer: [Lin Qihao (@lqh1106)](https://github.com/lqh1106).

μ-Bench evaluates a shared voice assistant serving several people in one continuous conversation. It tests whether the assistant preserves each person's requests, constraints, permissions and unfinished work as the conversation changes.

The release contains **800 cases**: **392 tool-free cases across 49 scenarios** and **408 tool-augmented cases across 51 scenarios**. Every scenario has eight combinations of group size, acoustic conditions and recipient comprehension. Cases, the online user simulator, audio protocols, tools, semantic judges and RICE computation are included.

[中文说明](README.zh-CN.md) · [Installation](docs/INSTALL.md) · [Agent setup](AGENTS.md) · [Protocol](docs/PROTOCOL.md) · [Evaluation](docs/EVALUATION.md) · [Models](docs/MODELS.md) · [Architecture](docs/ARCHITECTURE.md) · [Tool evaluation](docs/TOOL_EVALUATION.md)

## Run from a clone

Use the source checkout with an editable installation. Model weights and API credentials are supplied separately. The voice catalog, reference audio and frozen participant-to-voice bindings are included. Linux with Python 3.10–3.12 is supported for the runner; GPU backends use separate environments.

**Reproduction command — set the checkout location first:**

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

Deploy speech services and a model using [INSTALL.md](docs/INSTALL.md), then copy an example configuration to an ignored `local.*.json` file. Secrets are read from the environment, not embedded in the configuration.

**Reproduction command — one audio episode, after service setup:**

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

Omit `--case` to run the entire selected subset. Each case starts a fresh model session. Parallel lanes claim cases under a shared output root; they skip already completed cases and do not share conversation memory. A technical failure preserves evidence and advances to the next case. It is not recorded as task failure.

## Tracks

| Track | Input and measured behavior |
|---|---|
| Tool-free audio | Spoken registration, scoped initial source, overlapping users, native model speech |
| Tool-augmented audio | Audio interaction plus explicit attribution and business function calls |
| Tool-augmented text | Speaker-name-labeled text, streaming prefix interruption, real virtual-tool execution; no acoustic measurement |
| Serial ablations | Serial; serial + trusted identity; serial + trusted identity + structured semantics |

Included native integrations: MiniCPM-o, Moshi, Freeze-Omni, FLM-Audio, Voila, Raon-SpeechChat, PersonaPlex and Lychee-FD. Closed-provider integrations: OpenAI Realtime, Grok voice, Gemini Live and MOSS Live. Capability boundaries and separate model environments are listed in [MODELS.md](docs/MODELS.md). The text diagnostic adapter uses Qwen3.8-Flash by default.

## Results

A completed episode contains `score.json`, `rice_score.json`, the case and participant contracts, played-speech evidence, tool/attribution records, all judge stages and the final stereo `audio/both.wav`. Authorization is reported separately from Task Success. RICE reports Routing, Continuity, Isolation and Equity with the component denominators retained.

Results are published beneath the configured output root, organized by model, subset and condition. The publisher uses hardlinks on one filesystem and copies across filesystems; `formal` and `runs` therefore do not duplicate audio bytes when hardlinks are available. Automatic cleanup happens only after score, RICE and final stereo evidence pass checks. It removes redundant mono stems and played/oracle fragments, while retaining final stereo audio, ASR, trajectories, metrics and judge audits. Failed or incomplete evidence is retained.

## Repository layout

- `cases/`: complete materialized cases; `cases/ablations/<setting>/<case>` is shared across models.
- `runtime/core/`: shared event scheduling, media clock, user simulator, TTS clients and tool execution.
- `runtime/runners/`: tool-free, tool-augmented audio/text and ablation entry points.
- `runtime/ablations/`: one model-independent transformation and auxiliary-record design.
- `judges/`: shared scoring, tool-specific contracts and audio/text evidence distinctions.
- `adapters/providers/`: OpenAI Realtime, Grok, Gemini, MOSS and streaming-chat transports.
- `adapters/native/`: model-specific loaders, sparse backend overrides and compatibility code.
- `services/`: CosyVoice HTTP TTS, Faster-Whisper ASR, the ASR/text gateway and native model service.
- `assets/voices/`: catalog, licensed reference clips and frozen bindings.
- `environments/`: per-model dependency records and upstream requirements.
- `third_party/`: retained third-party source and licenses.
- `scripts/`, `tests/`: installation, dataset/protocol checks and regression tests.

Original benchmark code and case content use Apache-2.0. Third-party code, voice assets and model checkpoints retain their own licenses; see [NOTICE](NOTICE) and [THIRD_PARTY.md](docs/THIRD_PARTY.md).
