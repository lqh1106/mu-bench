# μ-Bench

### Benchmarking Multi-User Task Completion in Shared Interaction Voice Agents

[中文](README.zh-CN.md) · [Quick Start](#quick-start) · [Installation](docs/INSTALL.md) · [Models](docs/MODELS.md) · [Evaluation](docs/EVALUATION.md) · [Agent setup](AGENTS.md)

**Can a voice agent complete interdependent tasks for multiple users in one shared conversation?**

Personal assistants help one person get things done. Shared activities—planning a trip, coordinating a meeting or updating a family schedule—require an assistant to remember each person's requirements, check how new proposals affect existing plans and follow up on unfinished requests. **μ-Bench evaluates this shared work through reactive multi-user conversations and explicit task outcomes.**

[![A child interrupts an airport journey to request a park stop. The assistant checks the ETA, obtains the parent's approval and preserves the airport deadline. The lower panels show Attribution, Threading and Coordination.](assets/figures/intro-teaser.png)](assets/figures/intro-teaser.pdf)

*An illustrative shared interaction. Click the figure for the vector PDF.*

A parent needs to reach the airport **by 10:00**; a child asks for a park stop. The assistant must check the detour, obtain approval and preserve the deadline, while recognizing “Thanks, Dad!” as peer-directed speech and “Mm-hm” as a backchannel. μ-Bench diagnoses three connected responsibilities:

- **Attribution:** identify who is speaking, whom they address and whose request is being handled.
- **Threading:** maintain, update and resume each participant's unfinished tasks.
- **Coordination:** satisfy interdependent goals while respecting permissions, privacy and shared constraints.

## Benchmark at a glance

**20 domains · 100 base scenarios · 800 cases · full-duplex voice**

| Track | Scenarios | Cases | What establishes completion? |
|---|---:|---:|---|
| **Tool-Free** | 49 | 392 | Spoken answers, clarification, refusal and coordination |
| **Tool-Augmented** | 51 | 408 | Conversation plus actual virtual-tool execution and required business effects |

Scenarios cover travel, households, meetings, healthcare, education, public services and more. Each has **eight matched variants (2 × 2 × 2)**, varying group size (2–3 / 4–7), acoustic profile (clean / realistic) and user comprehension/follow-up demand (strong / weak). The [case catalog](cases/) supports comparisons that hold the scenario and two factors fixed while changing the third.

## How it works

**Case playbook → reactive users ⇄ tested agent and tools → recorded evidence → task and ATC scores**

Each case defines participants, their knowledge, goals, constraints and an event graph. An orchestrator releases events from actual conversation observations; a scriptwriter generates participant-specific utterances; a voice actor renders them using fixed reference voices. Agent speech and tool effects drive subsequent interruptions, clarifications and request revisions. Evaluation checks the resulting conversation and environment against the authored goals.

The repository includes cases, voice references, model adapters, speech-service implementations and judges. [Architecture](docs/ARCHITECTURE.md) explains the components; [Protocol](docs/PROTOCOL.md) specifies playback, interruption, logical-time pauses and information boundaries.

## Quick Start


### 1. Clone and validate

The runner supports **Linux, Python 3.10–3.12**. Install Git, `ffmpeg`, `libsndfile` and the build tools required by your selected backends. This checkout includes the voice catalog, **3,200 reference recordings** and fixed participant-to-voice bindings; model weights and API credentials are supplied separately.

**Reproduction commands — new checkout:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
git clone https://github.com/lqh1106/mu-bench.git "$MUBENCH_REPO"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

### 2. Deploy services and configure a model

Follow [Installation](docs/INSTALL.md) for complete service commands. Keep the runner, speech services and each native model in their own environments.

| Component | Included implementation | Configuration |
|---|---|---|
| User TTS | CosyVoice HTTP service | Voice catalog, TTS weights and `speech.tts_endpoint` |
| Assistant ASR | Faster-Whisper large-v3 | ASR weights and the gateway's ASR address |
| ASR/text gateway | Included HTTP router | Text API address/key and a local gateway key |
| Simulator and judge | OpenAI-compatible text API | `text.base_url`, `text.api_key`, `text.model` (default `gpt-6-luna`) |
| Tested agent | Native GPU service or cloud adapter | `sut` section: model, endpoint(s) and provider credentials |

**Reproduction commands — prepare local configuration:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp .env.example .env
cp configs/run.native.example.json local.native.json
cp configs/run.closed.example.json local.closed.json
```

Fill `.env` and your chosen local config using [INSTALL.md](docs/INSTALL.md). The templates read secrets from environment variables. Native checkpoint paths and separate environments are documented in [Models](docs/MODELS.md). MOSS provides [audio](configs/run.moss.example.json) and [tool-capable](configs/run.moss.tools.example.json) templates. Local configs, tokens, weights and results stay outside version control.

### 3. Run a smoke case, then scale

**Reproduction commands — native Tool-Free audio, after service setup:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

**Reproduction commands — Tool-Augmented audio with a configured cloud provider:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.closed.json --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1
```

After the run, open `formal/<case>/score.json` for the task verdict, `atc_score.json` for diagnostics and `audio/both.wav` to hear the conversation. These live under `output_root/<model>/<subset>/<condition>/`. [Output layout, full-subset runs and parallelism →](docs/USAGE.md)

## Supported models and tracks

| Integration | Tool-Free audio | Tool-Augmented audio | Identity/semantic ablations |
|---|:---:|:---:|:---:|
| MiniCPM-o | ✓ | — | ✓ |
| Moshi, Freeze-Omni, FLM-Audio, Voila, Raon-SpeechChat, PersonaPlex, Lychee-FD | ✓ | — | — |
| OpenAI Realtime, Grok voice, Gemini Live | ✓ | ✓ | — |
| MOSS Live | ✓ | ✓ | ✓ |

[Model deployment](docs/MODELS.md) provides checkpoint information, isolated environments and provider configuration. Availability and capacity depend on the configured endpoint.

**Capability ablations.** Three cumulative conditions preserve the task goals and source material: **serial turns → serial + identity → serial + identity + structured semantics**. The 1,176 transformed cases are shared across models; MiniCPM and MOSS implement the required auxiliary-input transports. [Settings and verified delivery →](docs/USAGE.md#capability-ablations)

**Text diagnostics.** An optional streaming text track uses speaker-name-labeled utterances, logical-time prefix interruption and the same virtual tools. It needs no speech services or GPU. The default tested chat model is `qwen3.8-flash`. [Text setup and interpretation →](docs/TEXT_TRACK.md)

## Evaluation

**Task Success** measures whether the case's applicable goals, required business effects, final state and privacy constraints are satisfied. **Authorization and caller-source violations are reported separately**, with the underlying records retained.

**ATC** explains the interaction capabilities behind the outcome:

| Axis | Meaning | Component mapping |
|---|---|---|
| **A — Attribution** | Speaker, addressee and request attribution | `A = Routing` |
| **T — Threading** | Keeping and resuming task threads | `T = Continuity` |
| **C — Coordination** | Boundary compliance and applicable multi-user service equity | `C = (Isolation + Equity) / 2`; when Equity is inapplicable, `C = Isolation` |

Scores use `[0, 1]`; inapplicable Equity is excluded, while an observed zero is included. ATC is computed per case from the preserved RICE components without another judge call.

Evaluation combines deterministic tool/state checks, semantic assessment, independent full review, conflict arbitration when needed and targeted source review. Judges assess the case's designed goals; simulator-added requests do not become mandatory goals. Complete trajectories can be judged offline without replaying the agent.

[Scoring and evidence flow](docs/EVALUATION.md) · [Tool evaluation contracts](docs/TOOL_EVALUATION.md) · [Outputs and automatic cleanup](docs/USAGE.md#outputs-and-cleanup)

## Documentation

| I want to… | Start here |
|---|---|
| Deploy TTS, ASR and the runner | [Installation](docs/INSTALL.md) |
| Configure or deploy a tested model | [Models](docs/MODELS.md) |
| Run a full subset, ablation or offline evaluation | [Usage](docs/USAGE.md) |
| Understand shared-channel interaction | [Protocol](docs/PROTOCOL.md) |
| Inspect task decisions and diagnostic scores | [Evaluation](docs/EVALUATION.md) |
| Run the streaming text track | [Text diagnostics](docs/TEXT_TRACK.md) |
| Locate components or automate installation | [Architecture](docs/ARCHITECTURE.md) · [Agent setup](AGENTS.md) |

## Citation and license

Author and maintainer: [Lin Qihao (@lqh1106)](https://github.com/lqh1106). Software citation metadata is in [CITATION.cff](CITATION.cff).

Original benchmark code and case content use [Apache-2.0](LICENSE). Third-party code, voice assets and external weights retain their respective licenses; see [NOTICE](NOTICE) and [Third-party components](docs/THIRD_PARTY.md).
