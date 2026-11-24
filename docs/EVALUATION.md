# Evaluation and evidence flow

## Evidence construction

The routine judge packet contains the authored task obligations, each participant's case-local name/public ID/internal ID/role/goal table, scoped initial assistant source, actually played user events, assistant ASR from played audio, delivered tool results and the real final environment state. Full internal tables are evaluator-only. They are not inserted into the tested model's conversation.

Evidence sources have stable IDs, timestamps, types and quoted content. Positive completion must cite the appropriate evidence type. A tool denial cannot prove successful execution. A spoken promise cannot prove a completed state change. Tool-free spoken answers do not require database writes. A simulator-added request is not a new mandatory benchmark goal.

## Semantic stages

1. **Primary evaluation** returns every goal, completion check, fact check, capability assertion and privacy finding.
2. **Independent full review** evaluates the complete schema from the case and evidence. It is not limited to the first judge's positive items and does not see that judge's answer.
3. **Conflict arbitration** receives anonymously ordered full reviews and original evidence. It resolves disagreements without preferring success, failure or a particular judge.
4. **Conditional source review** searches negative rationales for precise allegations that an assistant fact lacked an input source. Only on a hit does it add the preserved registration utterances, activation/phase prompts and actually delivered auxiliary records to a source-specific packet. It checks exact quoted sources and proposition direction, then revisits affected verdicts and downstream dependencies. Source support alone does not establish task completion.
5. **Validation and deterministic scoring** check schema coverage, evidence quotes/IDs, participant mappings, temporal dependencies and final effects. Unsupported or ambiguous evidence is retained for resolution rather than silently converted into a successful run.

The independent stages default to the same configured model family (`gpt-6-luna`) in independent contexts; this is not independence of model weights. All requests, responses, conflict lists, source checks and corrections are saved.

Ablation judges use the same base contract with condition-specific source search. Serial-only cannot cite routing or semantics it never received. Native auxiliary records require an actual prefill application event; MOSS records require all acknowledgment parts. Merely queued records do not establish delivery.

## Task Success and authorization

The primary Task Success is the conjunction of correct goal dispositions/completion, required business effects, required final state and privacy. Authorization violations are reported separately. The scoring output retains the underlying flags, denied/executed attempts and caller-provenance findings, so another aggregation policy can be reconstructed without replaying the conversation.

Separating authorization does not exempt a wrong business result, privacy leak, unsupported completion claim or unresolved user goal. Those can independently cause task failure.

## ATC: Attribution, Threading and Coordination

ATC presents the existing evidence-based component scores in three axes. It does not run another judge or change Task Success.

| ATC axis | Mapping | Interpretation |
|---|---|---|
| **A — Attribution** | `A = R` | Participant/addressee recognition, discourse provenance and return spoken anchors |
| **T — Threading** | `T = Continuity` | Maintaining, updating and resuming participant-specific task threads |
| **C — Coordination** | `C = (I + E) / 2` when E applies | Boundary compliance and applicable multi-user service equity |
| **C when E is absent** | `C = I` | Isolation alone; absent Equity is not zero-imputed |

`I` retains privacy, authorization and isolation evidence. `E` retains applicable fairness, group collaboration and quiet-input/floor-policy evidence. A numerical E of zero is applicable and participates in the two-component mean. Missing I leaves Coordination absent, rather than substituting E. All scores are in `[0, 1]`.

The public artifact is `atc_score.json`, containing `attribution_score`, `threading_score`, `coordination_score`, I/E values, the Coordination denominator (one or two), applicability and the original component units. It records the SHA-256 of `rice_score.json`. Raw RICE is retained for compatibility and audit: `routing_score`, `continuity_score`, `isolation_score`, and `equity_score` are unchanged. **ATC C means Coordination; the older RICE C means Continuity and maps to ATC T.**

Each original RICE axis averages its applicable component rates. Individual assertion/event-level records and denominators remain in `score.json` and trajectories. Both `rice_score.json` and its ATC wrapper are derived from the current judge/evaluator output, including `recognition_split`; no additional semantic judgment is needed to aggregate them. The older `interaction_score_rice` is retained as a legacy diagnostic, not renamed to an ATC average.

For dataset summaries, aggregate per-case ATC over the declared set of available results and retain its coverage. Compute each case's Coordination before averaging: a mean of aggregate I and E can differ when their applicable cases differ.

Text-track ATC retains `voice_comparable=false`: it measures name-labeled routing and logical interruption/threading, not voice identity recognition or physical overlap.

## Offline continuation

Complete interaction evidence can be judged again without replaying the tested model. Completed primary-stage artifacts can be reused by the checkpoint mechanism when compatible with the same request/evidence. An incomplete interaction is not made complete by a judge. New offline results are written to a separate `judge_resume_*` directory; original evidence and scores remain intact.

## Publication and cleanup

Publication requires a task verdict, required ATC axes and their source RICE components, and for audio a readable nonempty stereo final waveform. Raw judgments and receipts are retained. Redundant mono stems, oracle stems and played audio fragments are removed only after sealing; the final stereo recording, ASR and JSON provenance survive. Incomplete attempts are not automatically deleted.


The tool-augmented track applies the [business, caller, evidence and privacy contracts](TOOL_EVALUATION.md). Tool-free and ablation tracks use their corresponding evidence profiles.

### Judge retries and publication

A semantic retry receives copies of the original delivered-input receipts. Once a retry completes, its actual judge audit is adopted at the episode root while the previous failed audit remains in history. Publication requires the completed independent-review audit to cover every listed item. A score file alone is insufficient. Missing or incomplete audit evidence remains a technical recovery item, not an automatic task failure.
