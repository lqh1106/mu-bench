# Running experiments and inspecting results

For first-time installation and service deployment, start with [INSTALL.md](INSTALL.md) and the [README Quick Start](../README.md#quick-start).

## Selecting and scaling runs

Omit `--case` to run the full selected subset, or pass a base scenario name to select its eight variants. Each case starts a fresh session. Raise runner `--workers` and model-service session capacity together. All runners in one experiment must share an output root for case-claim locking. Each MOSS lane needs a distinct key endpoint, and a case keeps the same key throughout its session. See [model deployment and capacity](MODELS.md).

## Capability ablations

The three conditions preserve the task goals, constraints and initial source material:

| Condition | CLI value | Input change |
|---|---|---|
| Serial | `serial` | Sequential user turns without user overlap or real-time barge-in |
| Serial + identity | `serial_identity` | Additionally supplies trusted speaker, addressee, request-owner and thread records |
| Serial + identity + semantics | `serial_identity_semantics` | Additionally supplies a structured interpretation of the current utterance and already heard context |

Each condition contains 392 cases. The catalogs are shared across models. MiniCPM and MOSS have the auxiliary-input transports required for identity/semantic conditions. Auxiliary records do not grant permission or provide hidden reference answers; actual backend application acknowledgments establish delivery. Structured-input validation and adoption are audited. See [Protocol](PROTOCOL.md) for the delivery and generation rules and [Installation](INSTALL.md) for the complete compilation/run example.

## Outputs and cleanup

```text
outputs/<model>/<subset>/<condition>/
├── formal/<case>/
│   ├── score.json               # Task verdict and detailed component checks
│   ├── atc_score.json           # Attribution / Threading / Coordination
│   ├── rice_score.json          # Original R / Continuity / I / E components
│   ├── audio/both.wav           # Final stereo audio (audio track)
│   └── ...                      # Case, played speech, ASR, tools and judge stages
├── runs/<case>/attempt_001/      # Logs, receipts and preserved attempt evidence
└── claims/                      # Shared case locks
```

`output_root` is configurable. Different experiment versions should use separate roots. Complete cases are skipped on rerun; a valid task failure is retained as a result. Technical failures preserve evidence and allow the queue to advance. Complete trajectories can be [judged offline](INSTALL.md#offline-judging) without another model session; fresh offline outputs also include ATC.

Automatic cleanup is enabled in the example configs. It removes redundant mono/oracle audio and played fragments only after publication checks pass. Final stereo audio, ASR, case contracts, tool records and all judge stages remain. Same-filesystem publication uses hardlinks where possible. Incomplete attempts are retained; local TTS caches and attempt logs are not all removed by this policy.

## Inspecting a run

Read the task verdict and component checks in `score.json`, then inspect `atc_score.json` and its source `rice_score.json`. For audio runs, listen to the final `audio/both.wav` alongside ASR and played-event records. Tool runs also retain actual arguments, results, caller-source findings and final business state. Judge requests, reviews, arbitration and source checks explain how each verdict was reached.

[Evaluation](EVALUATION.md) defines evidence types, aggregation and offline continuation. [Installation: offline judging](INSTALL.md#offline-judging) gives the command to rejudge a complete trajectory; new results are written separately from the original evidence.

## Project map

| Directory | Purpose |
|---|---|
| [cases/](../cases/) | Task designs and shared ablation variants |
| [runtime/](../runtime/) | Simulator, clocks, scheduling and execution profiles |
| [adapters/](../adapters/) | Native model loaders and provider interfaces |
| [judges/](../judges/) | Evidence rules, reviews and underlying component scoring |
| [mubench/](../mubench/) | CLI, publication, offline evaluation and ATC wrapper |
| [services/](../services/) · [environments/](../environments/) | Deployment services and model dependency records |
| [assets/voices/](../assets/voices/) | Voice references, catalog and bindings |
| [configs/](../configs/) · [scripts/](../scripts/) · [tests/](../tests/) | Configuration, installation and validation |
| [third_party/](../third_party/) | Upstream source and original licenses |
