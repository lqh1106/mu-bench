# Model integrations

The runner and GPU model processes have separate Python environments. A model service holds causal state for a session and receives only revealed PCM. Every case starts a fresh session. Models retain their native default prompt on the tool-free track; no hidden oracle transcript or goals are sent to improve their answers.

| Public configuration name | Checkpoint / API | Service-specific integration |
|---|---|---|
| `minicpm` | `OpenBMB/MiniCPM-o-4_5` | Streaming prefill/generation; reference voice; mixed audio/text prefill for supported ablations |
| `personaplex` | `nv-community/personaplex-7b-v1` | Stateful Moshi-family codec/LM; NATF0 voice prompt |
| `raon` | `KRAFTON/Raon-SpeechChat-9B` | Stateful duplex decoding; explicit cache-length planning; Python compatibility source overlay |
| `lychee` | `HIT-TMG/Lychee-FD` | Owned HF realtime runtime, Step-Audio token2wav; source overlay |
| `moshi` | `kyutai/moshika-pytorch-bf16` | Stateful Mimi and LM streaming; pinned checkpoint revision in download helper |
| `freeze_omni` | `VITA-MLLM/Freeze-Omni` + `Qwen/Qwen2-7B-Instruct` | SocketIO companion plus PCM logical-service bridge |
| `flm_audio` | `CofeAI/FLM-Audio` | Stateful Mimi / FLM generation; real codec required |
| `voila` | `maitrix-org/Voila-autonomous-preview` + `maitrix-org/Voila-Tokenizer` | Official Homer Simpson reference; past-only codec context, 1-second input blocks; per-session debug state |
| `openai` | OpenAI Realtime | Played-frontier cancel/truncate handling, persistent send/receive state |
| `xai` | Grok voice | Native WebSocket audio and function-call outputs |
| `gemini` | Gemini Live | Native bidirectional audio, tool responses and session resumption |
| `moss` | MOSS Live | Session-scoped native audio, played-audio receipts, bounded same-session duration reconnection |
| `chat` | Qwen3.8-Flash by default | Streaming text tools, speaker-name headers, logical-time interruption |

The eight included native open-model integrations support the tool-free audio track. Their lack of native business function calling is not emulated by a hidden text planner. OpenAI/Grok/Gemini and MOSS have tool-capable adapters. MOSS uses native custom functions, not an external reasoning agent; it retains its session-scoped PCM/playback/reconnection behavior. Function results are acknowledged before continuation. Long wall-clock planner pauses use asynchronous function declarations. See `configs/run.moss.tools.example.json`. Serial audio ablation transports are implemented for MiniCPM and MOSS; another backend needs an actual auxiliary-context transport before running the identity/semantics conditions.

`environments/<model>/requirements.lock` records the model-side package stack, separate from the runner. Upstream license and requirements files are retained under `third_party/models`. CUDA/PyTorch wheels must match the target hardware. Freeze-Omni requires the companion process in `scripts/run_freeze_companion.sh`; the others run through `services/native/server.py`.

## Configuration

`configs/models/<model>.json` describes the loader and backend arguments using path placeholders. Copy `<model>.paths.example.json` to an ignored local file and fill the checkpoint/source/auxiliary paths. `mubench model-config` expands those paths and pins the copied loader SHA256. Loader helper hashes are checked through `SOURCE_MANIFEST.json`.

Weights are external to Git. The download helper applies the included checkpoint Python compatibility overlay after fetching weights. Its default branch is recorded; for a new benchmark release, pin `--revision` and retain the download receipt. Model license/access acceptance is handled by the user's Hugging Face account.

Lychee requires the Step-Audio-2-mini `token2wav` assets. PersonaPlex requires the NATF0 voice prompt distributed with its upstream voice pack. Paths for these are explicit in the model path template. Voila requires its separate audio tokenizer; Freeze-Omni requires its separate LLM.

## Concurrency

Start a new environment with one session per GPU, verify the model's own speech, then raise `--max-sessions` and runner `--workers` as VRAM and throughput allow. Sessions use separate state and output directories. Increase service and runner admission together. MOSS needs a distinct key endpoint per concurrent lane and retains that key for the complete session.

Independent runner processes must share the same output root to share case claims. Do not launch two experiments with separate claim roots and expect duplicate suppression. A case with a complete release receipt is skipped. Technical failures remain resumable evidence, rather than a false task-success denominator entry.

Provider transports are located in `adapters/providers/{openai,grok,gemini,moss,chat}`. Runtime profiles select task and evidence protocol, independently of model selection. All three ablation case catalogs are common across models; only auxiliary-input transport and its verified delivery differ. Runtime fingerprints are under `runtime/fingerprints/`, never inside a model-specific case copy.

## Seeduplex 3.0

Use [the Seeduplex configuration](../configs/run.seeduplex.example.json) and set `MUBENCH_SEEDUPLEX_KEY` to a Doubao Speech API Key. The explicitly pinned native model ID is `1.2.6.1`; the default output directory is therefore `outputs/1.2.6.1/<subset>/baseline/`. Both audio subsets use the same runner, simulator, case inputs and judge contracts as the other closed providers. No additional LLM decides its tools. See [Seeduplex protocol and qualification](SEEDUPLEX.md).
