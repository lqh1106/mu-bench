# Architecture

Cases define tasks and controlled conditions independently of the tested model. Model adapters implement the real provider input/output transport. Runners connect the two, while judges consume preserved evidence after interaction.

```text
cases/
  tool_free/<case>.json
  tool_augmented/<case>.json
  ablations/
    serial/<case>/{case,manifest}.json
    serial_identity/<case>/{case,manifest}.json
    serial_identity_semantics/<case>/{case,manifest}.json
runtime/
  core/                     shared clocks, simulator, tools and speech helpers
  runners/                  tool_free / tool_augmented_audio / tool_augmented_text / ablations
  protocols/                sparse task-specific execution changes
  ablations/                shared transformation, routing/semantic records and pacing
  fingerprints/             implementation pins for each realized ablation transport
judges/
  common/                   shared evaluator, evidence rules, reviews, source checks and ATC (with raw RICE components)
  tool_free/                tool-free semantic judging entry
  tool_augmented/           business/caller rules and audio/text semantic entries
  text/                     text evidence changes only
  ablations/                condition-specific source search and reassessment
adapters/
  providers/
    openai/                 GPT Realtime
    grok/
    gemini/
    moss/
    chat/                   streaming text diagnostic transport
  native/<model>/           loaders and backend-specific stateful-service overrides
services/                   ASR, TTS, gateway and native-model service launchers
```

## Model-independent ablations

There are 1,176 transformed cases: three conditions × 392 source cases. MiniCPM and MOSS use exactly these same files. A case manifest contains case, transformation and task-contract hashes. It does not contain a model's implementation fingerprint.

`runtime/fingerprints/minicpm.json` and `moss.json` pin the runtime/judge/provider files used by those transports. An implementation change invalidates its runtime pin without creating a different case design. Recompilation writes the shared case catalog once, followed by the separate implementation fingerprints.

## Shared code and sparse protocol overrides

Shared implementations live once under a common Python import namespace. A short source-path resolver selects only the genuine task/evidence differences: tool-free versus tool-augmented, audio versus text, baseline versus ablation. Package path ordering makes the selected override take precedence over the common implementation; regressions check those boundaries.

A model choice selects `adapters/providers` or a native endpoint, not another copy of all cases and judges. The tool-free audio runner is shared by native models, GPT Realtime, Grok, Gemini and MOSS. The MOSS-specific initial-source and reconnect behavior lives behind the provider selection. Extra ablation information still needs a real supported delivery API, so current auxiliary-context transports are MiniCPM and MOSS.

The per-model GPU backend directories contain only different native implementations and their loaders, not copied case generators or judges. The stateful HTTP service uses the shared core with the selected backend override.

## Outputs

The configured output root contains model-, subset- and condition-specific runs and accepted results. Preserved evidence includes case contracts, participant tables, delivered conversation, tool logs, judge stages, scores and final audio. See [Evaluation](EVALUATION.md) for evidence requirements and [Installation](INSTALL.md) for service configuration.
