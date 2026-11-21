# Seeduplex 3.0

This adapter targets the native Doubao Seeduplex 3.0 WebSocket API with explicitly pinned model `1.2.6.1`. It does not select the separate Doubao-Seed-Realtime-1.0 protocol or the older O/SC APIs.

- [Official API reference](https://docs.volcengine.com/docs/doubao-voice/end-to-end-real-time-voice-full-duplex-version?lang=zh)
- [Protocol and audio requirements](https://docs.volcengine.com/docs/doubao-voice/access-must-read?lang=zh)
- The playback-onset policy is also implemented in the Python demo linked from that API reference.

## Common benchmark contract

The ordinary Tool-Free and Tool-Augmented audio runners are used. Case goals, event triggers, participant TTS, initial materials, speaker registration, tool schemas, explicit public `caller_id`, simulator observations, task time limits and judges are unchanged. No external LLM selects or translates the agent's tool calls. The same evidence/authorization separation applies.

The runner delivers session instructions and participant introductions through the existing spoken protocol. The adapter sends exactly the shared runner's system prompt and tool declarations; it adds no roster, ASR hotwords, privileged roles, future event text, target answer or fabricated tool acknowledgment. Where the provider lacks an auto-response-disable switch, the request is audited as unsupported and actual registration replies remain evidence, as with Gemini.

The common cloud-provider interface is 8 kHz mu-law and 200 ms logical ticks. Only format conversion differs: incoming audio is decoded/resampled to the required mono PCM16 16 kHz and paced in approximately 20 ms packets; output PCM16 24 kHz is converted back to the common interface. No model-specific amplification, denoising or changed overlap is applied. Final evaluation and user planning observe ASR of the played output, not the provider's generated-but-unplayed text.

## Temporary pause versus interruption

**Simulator pause:** benchmark time and playback stop at the same frontier. The background receiver may keep collecting provider events into a bounded queue, but those events do not become played evidence or simulator observations during the pause. Computed tool results are flushed using their native protocol. Pauses shorter than one second send no input-state event. For a longer pause, `input_audio_mute.commit` marks the input gap; resumption sends `input_audio_unmute.commit` and continues with the next unsent audio. No PCM is replayed, no new session is created, and no reply is cancelled for planning.

The official documentation describes these events as input-gap keepalive, not as a freeze/resume inference API. No cloud inference-freeze guarantee is documented; this adapter does not implement or claim one. In the live five-second pause check, new provider events continued arriving. This is the same local logical-media guarantee used for other closed audio providers, not a claim of wall-clock or server-inference equivalence.

**Native interruption:** the official Seeduplex Python client clears its playback queue on `conversation.item.input_audio_transcription.started`. The adapter applies that native signal to only the unplayed queue/frontier. It does not use case speaker labels, ground-truth overlap, or text keywords to manufacture a cancel. Subsequent native audio continues to be processed. An explicit provider `response.canceled` additionally suppresses late output with the cancelled response ID, without clearing a newer response's audio.

No supported item-relative audio-history truncation API is documented for this endpoint. The adapter therefore does not invent `conversation.item.truncate`, inject an ASR transcript into cloud history, or claim the provider history is aligned to the local played prefix. Interruption/history uncertainty remains an audit diagnostic, as on Gemini; it does not silently change Task Success.

## Native tools and lifecycle

Each actual `response.function_call_arguments.done` call is passed to the ordinary benchmark tool executor with its original name, arguments and `call_id`. Duplicate identical call IDs are not re-executed; conflicting duplicates are technical errors. For a parallel call batch, all results are paired and returned together in one `conversation.item.create` event, with `role=tool` and `content[].type=input_text`. The provider then continues natively. No OpenAI-style `response.create` is sent.

The actual `session.created` ID establishes the session identity. Every case gets a fresh session; unexpected disconnects and provider errors remain technical errors, without stitching a replacement conversation into the same result. Shutdown sends `session.close` and records whether the actual `session.closed` acknowledgment arrived. Provider usage is retained when supplied.

## Configuration

Use a Doubao Speech API Key for `MUBENCH_SEEDUPLEX_KEY`; the simulator/judge text key is independent. Deploy the ordinary ASR, TTS and text gateway first, as in [Installation](INSTALL.md). The following are reproduction commands; fill the environment/configuration before running.

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp configs/run.seeduplex.example.json local.seeduplex.json
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.seeduplex.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

For Tool-Augmented audio, use the same config and the ordinary subset selector. No identity/semantic ablation transport is advertised for this provider.

## Qualification

Offline wire tests cover pacing, playback caps, native activity, cancellation frontiers, late response identities, pause/mute behavior, parallel tool batching, duplicate IDs, provider errors and usage. Live qualification records are kept outside the source checkout. A successful handshake alone is not evidence that a full benchmark episode or interruption test has passed. Inspect preserved audio, ASR and tool receipts before accepting a new deployment's scores.
