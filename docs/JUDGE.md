# Tool-augmented judge

The tool-augmented default is JT7. T-R5 revises the project-role, transit-accessibility, accessible-seat, and event-time cases; remaining tool cases retain T-R4 and use JT7. The published field-scoped business-value rules are described in [Business values](BUSINESS_VALUES.md). JT5 citation behavior is retained: quoted text is matched without case sensitivity. Reviews have up to three generation attempts. The third usable output is adopted with validation warnings retained, without rewriting verdicts. Independent review and conflict arbitration remain in place. Existing accepted results may carry an explicit compatibility label; this is not a fresh rejudgment. Tool-free JV4 is unchanged.

Business request-basis unknown handling (`business-basis-unknown-as-contradicted-v1`): a review's raw `insufficient_evidence` status, explanation and citations are retained. Its effective downstream status is `contradicted`, so it cannot satisfy a positive supported-call obligation and it no longer blocks the entire semantic evaluation. This is an explicit routing convention, not a claim that missing evidence proves a factual contradiction. Original primary/independent/arbitration responses are never overwritten. Caller/ACL unknown handling is unchanged.


## Separate text and speech endpoints

An optional `text.direct_base_url` in the run configuration routes the configured simulator/judge text model directly to a text endpoint accepting the same bearer credential. `text.base_url` continues to serve audio transcription requests. This avoids a speech gateway's shorter read timeout truncating long text judgments. The default route is retained when the option is absent. Native tested-model sessions, TTS settings, evidence, prompts and scoring rules are unaffected. The option is used by both `run` and offline `evaluate`; it also supplies the simulator's text client endpoint. Do not point audio requests at a text-only relay.
