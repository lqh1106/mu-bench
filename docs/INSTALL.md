# Installation and deployment

All commands below are reproduction commands. Set `MUBENCH_REPO` to the cloned checkout. Run long-lived services in separate terminals or a process manager. The benchmark runner, ASR, TTS and each tested model use separate Python environments.

## Runner

Linux and Python 3.10–3.12 are supported. Install system `ffmpeg`, `libsndfile1`, compiler/build tools and Git before Python dependencies. The benchmark runs from an editable source checkout.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

Create local secrets and copy a run template. A gateway key is local authentication for the speech/text router, not the upstream provider key.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
mkdir -p outputs/services
.venv/bin/python -c 'import secrets,pathlib; p=pathlib.Path("outputs/services/tts.key"); p.write_text(secrets.token_urlsafe(32)); p.chmod(0o600)'
cp configs/run.native.example.json local.native.json
cp configs/run.text.example.json local.text.json
```

Set `TEXT_API_KEY`, `TEXT_BASE_URL` and `MUBENCH_GATEWAY_KEY` in the local shell or an ignored `.env` file. The text endpoint must expose an OpenAI-compatible `/v1/chat/completions`, support the configured model, JSON responses, and streaming tools for the text track. Default simulator and judge: `gpt-6-luna`. Do not put real tokens in version-controlled examples.

## ASR

Faster-Whisper large-v3 receives actual played assistant audio. The included server exposes the audio-chat JSON interface expected by the runner and retains confidence/no-speech diagnostics. It does not replace actual speech with generated assistant text.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
python3 -m venv .venv-asr
.venv-asr/bin/python -m pip install faster-whisper numpy
.venv-asr/bin/python scripts/download_weights.py --model asr --destination "$MUBENCH_REPO/external/weights/asr"
FW_MODEL_PATH="$MUBENCH_REPO/external/weights/asr" FW_DEVICE=cuda FW_COMPUTE_TYPE=float16 bash scripts/run_asr.sh
```

## TTS

CosyVoice3 uses the included 3,200-voice catalog and frozen case/participant bindings. Reference recordings and binding hashes are validated before rendering. Its output is case-scoped and content-addressed.

The included CosyVoice source snapshot is under `third_party/models/CosyVoice/source`. Install its requirements in an isolated environment; install any submodule dependencies listed in that source's requirements in the same environment. The Matcha-TTS source is bundled when present in the frozen source tree. Obtain model weights separately.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
python3.10 -m venv .venv-tts
.venv-tts/bin/python -m pip install --upgrade pip
.venv-tts/bin/python -m pip install -r third_party/models/CosyVoice/requirements.txt
.venv-tts/bin/python -m pip install pydantic soundfile
.venv-tts/bin/python scripts/download_weights.py --model tts --destination "$MUBENCH_REPO/external/weights/tts"
COSYVOICE_REPO="$MUBENCH_REPO/third_party/models/CosyVoice/source" COSYVOICE_MODEL_DIR="$MUBENCH_REPO/external/weights/tts" TTS_TOKEN_FILE="$MUBENCH_REPO/outputs/services/tts.key" bash scripts/run_tts.sh
```

The service writes `outputs/services/tts.json`, containing endpoint discovery and the path to the local token file. For cross-node deployment set `TTS_BIND_HOST=0.0.0.0` and `TTS_ADVERTISE_HOST` to the service host. The runner must be able to read the endpoint and token path. Use your infrastructure's private networking/firewall.

## ASR/text gateway

The runner's text bridge sends both simulator/judge requests and ASR requests to one configured URL. The gateway routes audio requests to local ASR and text requests to the configured text API. It limits simultaneous requests and exposes queue metrics.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
export NO_PROXY=127.0.0.1,localhost
.venv/bin/python services/gateway.py --host 127.0.0.1 --port 18992
```

`TEXT_API_KEY`, `TEXT_BASE_URL`, `MUBENCH_GATEWAY_KEY` must already be set. Use the same gateway key in the runner's environment. ASR and TTS can share a GPU or use separate GPUs; they are external services, independent of the tested model.

## Native model service

For each model, use [MODELS.md](MODELS.md) and its path template. The following MiniCPM example downloads its checkpoint, applies the included compatibility code, constructs a local config, and serves persistent sessions. Model weights remain outside Git.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
bash scripts/setup_model.sh minicpm
.venv/bin/python scripts/download_weights.py --model minicpm --destination "$MUBENCH_REPO/external/weights/minicpm"
cp configs/models/minicpm.paths.example.json local.minicpm.paths.json
.venv/bin/python -m mubench model-config --model minicpm --paths local.minicpm.paths.json --output outputs/services/minicpm.config.json
.venv/bin/python -c 'import secrets,pathlib; p=pathlib.Path("outputs/services/minicpm.key"); p.write_text(secrets.token_urlsafe(32)); p.chmod(0o600)'
CUDA_VISIBLE_DEVICES=0 .venv-minicpm/bin/python services/native/server.py --model minicpm --config "$MUBENCH_REPO/outputs/services/minicpm.config.json" --output-root "$MUBENCH_REPO/outputs/backends/minicpm" --token-file "$MUBENCH_REPO/outputs/services/minicpm.key" --endpoint-file "$MUBENCH_REPO/outputs/services/minicpm.json" --host 127.0.0.1 --advertise-host 127.0.0.1 --port 19000 --max-sessions 1
```

The native endpoint initially describes the service and model quantum. Health and the first session establish actual readiness; endpoint-file existence alone is not readiness. Freeze-Omni additionally starts `scripts/run_freeze_companion.sh` with its checkpoint/LLM paths before the native service. See each model's explicit auxiliary paths in `configs/models`.

Then run the audio episode from README. Increasing concurrency requires both an adequate model-service session limit and runner `--workers`. For multiple GPUs, add independent endpoint files to `sut.endpoints`.

## Closed providers and MOSS

Copy `run.closed.example.json`, set `provider` to `openai`, `xai` or `gemini`, and use your actual endpoint/model/key environment. `run.moss.example.json` uses one endpoint file per key. Copy `moss.endpoint.example.json` to ignored local discovery, resolve the credential path absolutely, and store only the key in that mode-600 file. MOSS keys stay fixed inside a session. No account credentials are bundled.

## Text diagnostic run

Set the text API in `local.text.json`; the default tested model is `qwen3.8-flash`. No GPU, ASR or TTS is used. The same online simulator and tool judge are used with text-specific evidence types. See [TEXT_TRACK.md](TEXT_TRACK.md) for interruption scheduling and its measurement boundaries.

## Ablations

The release includes compiled serial/identity/semantic conditions for MiniCPM and MOSS. If editing public cases or scoring code, recompile before running; stale fingerprints are rejected.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
.venv/bin/python scripts/compile_ablations.py
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --ablation serial_identity --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

## Offline judging

Use an existing complete evidence directory and a current local text configuration. No tested-model session starts. The new judge result is written under `judge_resume_*`; keep it separate until you choose to publish it.

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
.venv/bin/python -m mubench evaluate --config local.native.json --subset tool_free --provider native --transport audio --case-dir "$MUBENCH_REPO/outputs/minicpm/tool_free/baseline/formal/airline_04_gate_announcement_collision__g-small__a-clean__c-strong"
```
