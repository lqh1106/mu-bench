# Streaming text diagnostic track

The text adapter uses the same case, participant-isolated `OnlineUserSession` writer/critic, candidate caps, event graph, closure planner, virtual environment and semantic judge as the current audio tool track. It omits synthesis and acoustic processing. User turns are labeled with their registered display name; business tools still use public participant codes.

Delivered assistant characters advance the logical clock using a declared estimate: CJK characters count as one token-equivalent, other characters as one quarter, at four token-equivalents per logical second. User utterance duration uses the same estimate. These are scheduling estimates, not a tokenizer measurement or real speech latency.

At each delivered character the event graph can release an interruption. The active stream is closed, the delivered prefix is committed to history, and the interrupting user utterance is appended. Buffered suffixes and future chunks are discarded. Reasoning-only fields are never spoken or committed. A partial function call does not execute. A complete function call executes once and its actual effect persists when a later user changes the request.

The transcript and `stream_audit.json` distinguish delivered prefixes, interrupted streams, discarded buffered suffixes and incomplete tool calls. Tool results and attribution receipts are real outputs from the virtual environment, not prose claimed by the tested model.

The reported score is a text diagnostic result. It cannot establish voice recognition, acoustic robustness, true overlap handling or cloud inference pause behavior.

**Reproduction command — after setting `TEXT_API_KEY` and the endpoint in `local.text.json`:**

```bash
export MUBENCH_REPO="$HOME/mubench"
cd "$MUBENCH_REPO"
.venv/bin/python -m mubench run --config local.text.json --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1
```

For a full run, omit `--case`. Per-case evidence includes `history.json`, `stream_audit.json`, `pause_ledger.json`, planner sessions, tool log, caller provenance, business-request basis and all judge stages. No WAV or ASR output is fabricated for this track.
