# μ-Bench

**Evaluating shared voice assistants in continuous multi-user conversations.**

[中文](README.zh-CN.md) · [Installation](docs/INSTALL.md) · [Protocol](docs/PROTOCOL.md) · [Evaluation](docs/EVALUATION.md) · [Models](docs/MODELS.md) · [Agent setup](AGENTS.md)

Author and maintainer: [Lin Qihao (@lqh1106)](https://github.com/lqh1106).

μ-Bench tests whether a voice assistant can complete tasks for several people sharing one conversation. Users interrupt, speak to one another, revise earlier requests and return to unfinished work. The assistant must keep track of whose request it is handling, which constraints still apply, and who may authorize a change.

The repository includes **800 cases across 100 scenarios**, an online user simulator, reference voices, native audio adapters, virtual business tools, evidence-based judges, and reproducible result artifacts. Supply model weights or API credentials and deploy the included ASR/TTS services to run the audio benchmark.

## What does a shared assistant need to do?

Consider a reservation conversation: Avery asks to update a reservation to eight people; Blake corrects the headcount to six; Avery returns to confirm the revised request. The assistant must reconcile these turns, identify the requester of each business action, and complete the correct final update. The environment records both successful changes and rejected calls.

In a tool-free airport conversation, Avery asks about a gate and mobility assistance while another person announces a different flight's gate. A third traveler adds a separate itinerary. The assistant must keep the flight records, speakers and requests distinct while continuing the conversation.

Tasks cover travel and hospitality, shared homes and family coordination, meetings, education, healthcare, customer support, public services, and other everyday shared activities.

## Cases and controlled conditions

| Subset | Scenarios | Cases | Completion evidence |
|---|---:|---:|---|
| Tool-Free | 49 | 392 | Actual spoken answers, case-defined goals, constraints and privacy |
| Tool-Augmented | 51 | 408 | Spoken interaction, real virtual-tool results, required business effects and final state |
| **Total** | **100** | **800** | |

Each scenario has **eight variants: 2 × 2 × 2**.

| Controlled variable | Values in case IDs |
|---|---|
| Group size | `g-small`, `g-large` |
| Acoustic condition | `a-clean`, `a-realistic` |
| User comprehension | `c-strong`, `c-weak` |

For example, `airline_04_gate_announcement_collision__g-small__a-clean__c-strong` identifies one concrete variant. All case definitions are materialized under [cases/](cases/).

## Quick start

### 1. Clone and validate

Use Linux with Python **3.10–3.12**. Install Git, `ffmpeg`, `libsndfile` and the build tools required by your chosen backends. The runner and model/speech services use separate Python environments.

**Reproduction commands — new checkout:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
git clone https://github.com/lqh1106/mu-bench.git "$MUBENCH_REPO"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

The release includes a **3,200-voice catalog**, reference recordings and fixed participant-to-voice bindings. Model checkpoints and API credentials are configured separately.

### 2. Configure speech, models and credentials

Follow [Installation](docs/INSTALL.md) to start:

| Component | Included implementation | Role |
|---|---|---|
| TTS | CosyVoice HTTP service | Render simulated users in their assigned voices |
| ASR | Faster-Whisper large-v3 | Transcribe the assistant audio that was actually played |
| ASR/text gateway | Local HTTP gateway | Route audio to ASR and simulator/judge requests to the text API |
| Tested model | Native service or provider adapter | Maintain one fresh, independent session per case |

The simulator and judge default to `gpt-6-luna`. Choose the model and endpoint in your local configuration. Model-specific checkpoints, environments and auxiliary assets are documented in [Models](docs/MODELS.md).

**Reproduction commands — create local configuration files:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp .env.example .env
cp configs/run.native.example.json local.native.json
cp configs/run.closed.example.json local.closed.json
```

Fill the local environment variables and endpoint paths following [INSTALL.md](docs/INSTALL.md). Configurations support environment references for secrets. `local.*.json`, `.env`, local tokens and generated outputs are ignored by Git. MOSS has separate [audio](configs/run.moss.example.json) and [tool-capable](configs/run.moss.tools.example.json) templates.

### 3. Run one case

Start with one session after the services and local configuration are ready.

**Reproduction commands — tool-free audio with a native model:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

**Reproduction commands — tool-augmented audio with a configured closed provider:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.closed.json --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1
```

Omit `--case` to run the entire selected subset. A base scenario name passed to `--case` selects its eight variants. Increase `--workers` together with the model service's supported session capacity. Independent runners must share the same output root for case-claim locking. MOSS requires a distinct key endpoint for each lane; each session keeps its assigned key.

## Model coverage

| Integration | Tool-Free audio | Tool-Augmented audio | Identity/semantic ablation transport |
|---|:---:|:---:|:---:|
| MiniCPM-o | ✓ | — | ✓ |
| Moshi, Freeze-Omni, FLM-Audio, Voila, Raon-SpeechChat, PersonaPlex, Lychee-FD | ✓ | — | — |
| OpenAI Realtime, Grok voice, Gemini Live | ✓ | ✓ | — |
| MOSS Live | ✓ | ✓ | ✓ |

Native loaders and model-specific dependency records are included for all eight open models. Closed-provider adapters use their native streaming interfaces. Tool calls execute against the benchmark environment with explicit public participant codes and per-call `caller_id`; attribution receipts acknowledge storage without revealing correctness.

See [Models](docs/MODELS.md) for checkpoint names, deployment paths and capability boundaries, and [Protocol](docs/PROTOCOL.md) for registration, input delivery, interruption and simulator-computation pauses.

## Serial, identity and semantic ablations

Three nested conditions use the same underlying task goals, constraints and business rules:

| Condition | CLI value | Input change |
|---|---|---|
| Serial | `serial` | Present user turns sequentially, removing user overlap and real-time interruption requirements |
| Serial + identity | `serial_identity` | Additionally supply speaker, addressee, request-owner and thread information |
| Serial + identity + semantics | `serial_identity_semantics` | Additionally supply a structured interpretation of the current user utterance and already heard context |

The **1,176 ablation cases are model-independent**: three conditions × 392 tool-free cases. Supported transports currently deliver the auxiliary information to MiniCPM and MOSS. Case files are shared across models; runtime fingerprints track the implementation separately. Structured-input generation and any final-candidate adoption are audited under the [protocol](docs/PROTOCOL.md).

**Reproduction commands — one identity-assisted episode after native service setup:**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --ablation serial_identity --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

## Evaluation

The runner preserves delivered user speech, played assistant speech and its ASR, tool outputs, participant mappings, and the authored task contract. The semantic judge performs **primary evaluation → independent full review → conflict arbitration when needed → conditional source review**. Evidence validation and deterministic checks precede publication.

- **Task Success** checks required goals, business effects where applicable, final state and privacy. Authorization violations are reported separately and retained for alternative aggregations.
- **Simulator-added requests** do not become additional mandatory benchmark goals.
- **Tool-free completion** is judged from the required spoken behavior; it does not require an unavailable database write.
- **Tool-augmented completion** separates spoken reports from actual execution and business state. Execution history remains available for process and authorization checks.

| RICE axis | What it measures |
|---|---|
| **R — Routing** | Speaker/addressee attribution, request provenance and return anchors |
| **C — Continuity** | Maintaining and resuming the correct task threads and constraints |
| **I — Isolation** | Privacy, authorization and separation between participants |
| **E — Equity** | Applicable fairness, group collaboration and quiet-input handling |

RICE scores are in `[0, 1]`. Available component rates are aggregated per axis; nonapplicable components are left absent. Item-level evidence and denominators remain in the saved results. See [Evaluation](docs/EVALUATION.md) and [Tool evaluation](docs/TOOL_EVALUATION.md) for exact contracts and evidence flow.

## Outputs and offline evaluation

With `output_root` set to `outputs`, results are organized as:

```text
outputs/<model>/<subset>/<condition>/
├── formal/<case>/                # Complete, accepted result
│   ├── score.json
│   ├── rice_score.json
│   ├── audio/both.wav            # Final stereo recording
│   └── ...                      # Case, ASR, traces and judge evidence
├── runs/<case>/attempt_001/      # Attempt logs, receipts and evidence
└── claims/                      # Shared case locks
```

Every case starts a fresh conversation. Completed cases are skipped on subsequent runs. A technically valid task failure is retained as a scored result; technical errors preserve their evidence separately and allow the queue to advance.

Automatic cleanup is enabled in the example configurations. After score, RICE and final stereo audio pass checks, it removes redundant mono stems and audio fragments. It retains the final recording, ASR, trajectories, case contracts and judge audits. `formal` uses hardlinks where possible, so publishing a result does not duplicate audio bytes on the same filesystem. Incomplete evidence is retained.

Complete interaction evidence can be evaluated offline without starting another tested-model session. The CLI writes a separate judge continuation result; see [Offline judging](docs/INSTALL.md#offline-judging).

## Optional text diagnostic track

The streaming-chat adapter uses speaker-name-labeled turns, the online simulator and virtual tools. A logical-time interruption commits the delivered text prefix, discards the unread suffix and introduces the interrupting utterance. Complete tool calls have real environment effects; incomplete calls do not execute.

The default tested chat model is `qwen3.8-flash`. Configure [run.text.example.json](configs/run.text.example.json) as `local.text.json` with an endpoint serving your selected models. No ASR, TTS or GPU is required for this track. Its metrics are labeled as text diagnostics. See [Text track](docs/TEXT_TRACK.md) for the timing estimate and evidence format.

## Repository guide

| Path | Contents |
|---|---|
| [cases/](cases/) | Complete tool-free, tool-augmented and shared ablation cases |
| [runtime/](runtime/) | Shared simulator, clocks, scheduling and task-specific runners |
| [adapters/](adapters/) | Native model loaders and provider transports |
| [judges/](judges/) | Evidence construction, semantic reviews, deterministic scoring and RICE |
| [services/](services/) | ASR, TTS, gateway and native model services |
| [assets/voices/](assets/voices/) | Voice catalog, reference recordings and fixed bindings |
| [configs/](configs/) | Run templates and model-path templates |
| [environments/](environments/) | Model-specific dependency records |
| [third_party/](third_party/) | Bundled upstream source and original licenses |
| [scripts/](scripts/) · [tests/](tests/) | Setup helpers, validation and regression checks |

For automated installation, start with [AGENTS.md](AGENTS.md). The [architecture guide](docs/ARCHITECTURE.md) explains how cases, protocols, adapters and judges connect.

## Citation and license

Software citation metadata is available in [CITATION.cff](CITATION.cff).

Original benchmark code and case content are licensed under [Apache-2.0](LICENSE). Third-party code, voice recordings and externally obtained model weights retain their respective licenses. See [NOTICE](NOTICE) and [Third-party components](docs/THIRD_PARTY.md).
