# Interaction protocol

## Common episode lifecycle

1. Load the materialized case and validate its identity, goal and tool contracts. Start a fresh tested-model session and one isolated simulator context per participant.
2. Deliver the assistant name, registration listening rule and per-utterance attribution rule in the first registration audio, followed by spoken self-introductions. These session rules are not duplicated in system text. No provider receives participant-name or participant-code ASR hotwords. Enroll voices through spoken self-introductions. Introductions state the public participant code, name, role and disclosed permissions. Internal oracle IDs are never substituted for public IDs in model-visible tool arguments.
3. Deliver the authored, scoped initial assistant source. It is not synthesized from future events, expected answers, private goals or hidden ACL data. Audio-only native backends receive a TTS readout after registration and before the formal task. Providers with initial instruction support receive the same authored content through that interface.
4. Play activation and the first request. The event graph schedules later speech from actual observations, completed events and explicit fallbacks. The simulator sees only what its participant may know and hear.
5. Advance one causal media tick at a time. Only newly revealed user PCM crosses the input boundary. Only played assistant PCM enters ASR evidence and simulator observations. Generated text and buffered future audio are not treated as heard speech.
6. End through the conversation completion gate, the task logical limit, or bilateral silence. A 600-second bilateral-silence completion requires both tracks idle, drained output and no pending tools/responses/protocol activity; silence is not model task failure. A technical interruption remains a technical result.
7. Preserve evidence, evaluate and publish. Default task logical limit: 1,800 seconds; registration/initial-source phase is separate. The wrapper imposes a 6,000-second episode wall-time budget.

## Pausing and interruption

Simulator computation freezes the local logical clock. Stateful native services advance only on synchronous media requests, so computation stops at the acknowledged boundary. Cloud providers may continue producing data during local planning: the adapters retain queues and playback accounting; this is not a claim that cloud inference was frozen.

A user interruption is an event in the conversation, separate from the simulator's computation pause. OpenAI's adapter maintains cancellation/truncation relative to played audio; native providers use their actual interruption controls and acknowledgments. The provider implementations and their audits are included rather than replaced with a common artificial reset.

MOSS reconnects only a recognized duration-limit closure to the same session, same key and session memory, retaining received audio and playback acknowledgments. New cases always use new identities/sessions. Other disconnects remain technical failures.

## Tool-free tasks

The assistant communicates an answer, correction, refusal, clarification or handoff. No business tool execution or external delivery receipt is required unless the case actually exposes such a tool. Registration/attribution records do not count as business actions. Native models without function calling are evaluated from their actual spoken behavior.

## Tool-augmented tasks

After a user utterance, the model reports attribution once before answering or calling business tools. There is no required spoken preface. `report_speaker_attribution` returns `{"recorded": true}` for a syntactically valid report: this acknowledges receipt and never discloses whether the report matches the oracle.

Each business tool independently requires `caller_id`: the registered public code of the person whose request the operation carries out. Use `p01`, `p02`, etc., not names or internal roles. The operation subject is a separate parameter. The environment checks the declared caller's ACL; it does not use the most recent attribution report to silently correct the caller. The private judge checks whether the declared caller is supported by requests/consent delivered before that call.

Tool outputs translate participant references back to public codes. Resource identifiers tolerate approved template/format variations, while preserving information-bearing digits and distinguishing resources. A malformed or unauthorized operation is recorded and returned as an error; the model can correct and call again.

Final state comparison projects explicitly enumerated business fields, excluding pure audit history and internal transaction sequence IDs. Scope, affected participants, values and true business state remain checked. Process violations are retained separately.

## Request revision and triggers

A lawful first operation followed by a valid revision is allowed. If the assistant waits, combining the original request and later objection into one correct operation is also allowed. The first operation is not a prerequisite for task success.

Objection events can trigger from an actual matching tool result or an explicit elapsed-time fallback. Wording must remain valid on the path actually taken. Queries are checked as data dependencies before the decisions or claims relying on them, not as arbitrary requirements to beat an interruption timestamp. Actual delivered requests/consents establish the temporal basis; a fixed event name alone does not.

## Controlled conditions

Each base scenario crosses small/large participant groups, clean/realistic audio and strong/weak recipient comprehension. The latter changes recipient language/repair requirements, not the tested assistant's model settings. Three cumulative audio ablations preserve task goals and source material: serial turns; serial + trusted routing; serial + trusted routing + faithful structured user input. Auxiliary records carry no new permission grant or hidden reference answer. Applied backend acknowledgments, rather than queued plans, establish delivery.

## Audio session setup and enrollment completeness

Session setup instructions are a separate narrator audio phase before participant introductions. The narrator is not a participant and does not consume an enrollment slot. Each enrolled participant then receives a complete introduction in their assigned voice, including their name, public participant code, role and authored permissions. Canonical participant codes remain `p01`, `p02`, etc.; speech spells out the letter and individual digits.

Registration speech is synthesized in short, punctuation-aligned segments. Segment boundaries, intended text, PCM hashes and participant timing are preserved in `registration_audio_segments.json`; separate session setup provenance and submission progress are preserved in `session_protocol_audio.json`. Submission receipts attest audio delivery, not word-level intelligibility; actual audio must be checked when validating an endpoint. These instructions are not duplicated in system text, and Grok receives no participant-name transcription hotwords. The ordinary conditional source review can retrieve the submitted setup audio transcript when resolving a factual-source allegation.

### Structured-input retry policy

The semantic-input ablation attempts at most three parses of each actual utterance. The next generation receives the previous candidate and the full parser or independent-review rejection, including its reason. The third candidate is adopted without another independent review, even if its validation fails; this is recorded as a forced adoption, never as a passed check. If no JSON object is available, the original utterance is relayed verbatim in a minimal record. HTTP or transport failures are not relabeled as valid parses. Per-utterance audits preserve the candidate, failed checks, feedback and adoption policy. Case goals and judge obligations are unchanged.

### Provider admission failures

HTTP 402 and recognized billing/authentication failures stop new case admission for that CLI batch. Cases already running finish naturally. The batch saves its reason and unstarted case IDs in `admission_status.json`; unstarted cases do not create attempts or count as task failures. Ordinary transient case errors retain the existing continue-to-next-case behavior. After restoring provider access, rerun the selected batch to resume; published cases are skipped. Shared credentials across separately launched CLI processes require a coordinator to honor the same block reason.
