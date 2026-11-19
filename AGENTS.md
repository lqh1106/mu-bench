# Agent setup guide

This checkout is an independent benchmark. Configure local model/speech services and API secrets; do not import a separate experiment directory or copy its results into a new run.

1. Read README.md, docs/INSTALL.md, docs/PROTOCOL.md and docs/EVALUATION.md.
2. Install the editable runner with scripts/bootstrap.sh. Use a separate virtual environment for ASR, TTS and every GPU backend.
3. Validate all cases and voice assets. Every path in configs/examples must be resolved locally; secrets use environment references.
4. Download the chosen checkpoint under external/. Apply the included checkpoint-code compatibility overlay. Preserve upstream license terms.
5. Start ASR, TTS, the ASR/text gateway and the model service. Check health; a stale endpoint file is not service readiness.
6. Run one case first. Inspect the actual final stereo audio, assistant ASR, delivered user events, tool results and judge evidence. Confirm there are no technical errors masquerading as task failures.
7. Scale with --workers and model --max-sessions. All lanes targeting the same experiment share one output root for claim locking. One MOSS key endpoint per lane; never change a key mid-session.
8. For tool cases, preserve explicit caller_id public codes. Do not expose private participant roles/goals/ACL oracle mappings to the tested model. Attribution acknowledgments disclose receipt, not correctness.
9. Keep evidence on errors. Only complete conversations qualify for evidence-only judging. Do not fabricate successful exit codes or replay partial tool streams.
10. Auto-cleanup is score/ATC/source-RICE/final-stereo gated. Keep final audio, ASR, trajectories, case contracts, judge requests/responses and receipts. Never delete incomplete evidence merely because a queue advances.

Use the public protocol identifiers produced by this release. Experimental histories and private deployment paths are not needed. Do not commit local keys, endpoint token files, logs, model weights or generated result trees.

Recompile the shared ablation catalog and separate runtime fingerprints if a case/scoring implementation changes. Run scripts/validate_release.py, scripts/check_entrypoints.py, the unit tests and tool revision/caller/transport fixtures after changing protocol code. Synthetic fixtures test mechanisms; they are not model evaluations.

ATC is the public diagnostic format: A=R, T=Continuity, C=(I+E)/2, or C=I when E is absent. Keep raw RICE components and applicability; never zero-impute missing Equity. New finalized and offline-evaluated outputs include atc_score.json without rejudging merely for aggregation.
