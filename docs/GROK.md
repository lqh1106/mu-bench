# Grok Voice

Use `provider: "xai"`, `transport: "audio"` and an explicit model version. New runs use `grok-voice-3.0`; the adapter does not select the floating `grok-voice-latest` alias or fall back to another model. Set the WebSocket URL to `wss://api.x.ai/v1/realtime` and supply the API key through the private run configuration.

The adapter uses native 8 kHz G.711 mu-law input/output and server VAD. Session instructions, participant introductions and initial scene materials follow the shared audio protocol. Function declarations expose the same public schemas as the other tool-capable providers. Results retain their original call IDs. The shared continuation scheduler waits for result receipts and an idle response boundary, coalesces continuation requests, and does not replay business actions after a response-creation conflict.

Registration silence is requested through the shared spoken instructions. The adapter does not claim that undocumented `create_response` or `interrupt_response` flags disable native replies. Actual enrollment replies remain evidence. No participant vocabulary hints or additional per-tool task coaching are injected.

A simulator-computation pause freezes local media progress. A bounded background receiver keeps consuming network events; it does not play audio, execute tools, request new replies, or alter cloud history during the pause. Buffered events are processed in order when media resumes. This is not a cloud inference pause. Queue overflow or an unrecoverable connection failure is a technical error rather than a task failure; the adapter does not replace the lost session with an empty conversation.

Real user interruptions use native server VAD and the common item-relative played-audio ledger. Unplayed late audio is suppressed. Truncate requests are deduplicated per connection by item ID, content index and played position. A pending or confirmed target creates no additional request or wait. Only a matching server acknowledgement confirms that target; a later response does not release the wait, and late acknowledgements remain matchable. A missing acknowledgement after 10 seconds of advancing media time is a recorded technical error; simulator-computation pauses do not consume that deadline. Episode finalization also waits for queued provider events and pending tool-result/continuation state.

References: [xAI speech-to-speech guide](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech), [Voice API reference](https://docs.x.ai/developers/rest-api-reference/inference/voice).
