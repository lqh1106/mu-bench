"""Single-session user planning. No speculative responses or replay prefixes."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from typing import Any

from experiments.tau_multiuser_voice.event_graph import EventGraphExecutor
from experiments.tau_multiuser_voice.iterative_replay import ModelTailEventOrchestrator, TailDecision
from experiments.tau_multiuser_voice.models import (
    ObservationKind, ScriptedEvent, TriggerKind, TriggerSpec, UserAction,
    UserProposal, WorldObservation,
)
from experiments.tau_multiuser_voice.sessions import PersistentUserSession
from experiments.tau_multiuser_voice.simulator import MultiUserVoiceSimulator
from experiments.tau_multiuser_voice.simulation_contract import (
    CONTRACT_VERSION, SimulatorInvalid, UserCandidate, actor_context,
    closure_public_context, compile_event_contract, critique_candidate, critique_closure, public_profile,
    validate_candidate, writer_prompt,
)

PROTOCOL_VERSION = "mu-single-session-logical-v14.0"
PROBE_FIELDS = {"probe_complete_assistant_response", "probe_assistant_fragments",
                "probe_assistant_intervals_ms", "probe_task_media_end_ms"}


def causal_observation_payload(observation):
    if any(getattr(observation, key) for key in PROBE_FIELDS):
        raise ValueError("speculative Probe evidence is forbidden in online planning")
    return observation.model_dump(mode="json", exclude=PROBE_FIELDS)


class LogicalPauseLedger:
    """Host latency is recorded separately; it never creates media samples."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def planning(self, sim_time_ms: int, kind: str = "user_planner_and_tts"):
        started = time.monotonic()
        record = {"logical_start_ms": sim_time_ms, "logical_end_ms": sim_time_ms,
                  "kind": kind, "wall_started_unix": time.time(), "status": "running"}
        try:
            yield
            record["status"] = "completed"
        except BaseException:
            record["status"] = "error"
            raise
        finally:
            record["wall_elapsed_ms"] = (time.monotonic() - started) * 1000
            self.records.append(record)


class OnlineUserSession(PersistentUserSession):
    """Retain one private LLM conversation per person, with causal prompts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.contract_audit = []
        self.hearing_intervals = [[0, None]] if self.profile.present_initially else []
        self.public_evidence_provider = None

    def record_presence(self, present, at_ms):
        if present and not any(b is None for _, b in self.hearing_intervals):
            self.hearing_intervals.append([at_ms, None])
        elif not present and self.hearing_intervals and self.hearing_intervals[-1][1] is None:
            self.hearing_intervals[-1][1] = at_ms

    def _system_prompt(self) -> str:
        return json.dumps({"schema": CONTRACT_VERSION,
            "your_profile": public_profile(self.profile),
            "rules": ["Control only this person, preserving their private goals and knowledge.",
                "This is one ongoing conversation, not a Probe or retrospective replay.",
                "Only supplied audible committed content has happened; never infer an unplayed suffix.",
                "Perform the USER event, not the assistant's expected answer or task completion.",
                "Never assume that a requested or promised action has already succeeded.",
                "The logical clock is paused during planning. Wall-clock delay is not conversational silence.",
                "Return one JSON user proposal, no prose. Do not reveal benchmark metadata."]}, ensure_ascii=False)

    def _proposal_prompt(self, observation, event) -> str:
        return writer_prompt(self._contract_context(observation, event),
                             alignment_ms=self.timing_alignment_ms)

    def _contract_context(self, observation, event):
        causal_observation_payload(observation)  # reject retrospective Probe data
        public = self.public_evidence_provider(observation) if self.public_evidence_provider else None
        return actor_context(self.case, self.profile, self.runtime, observation,
            compile_event_contract(self.case, event), hearing_intervals=self.hearing_intervals,
            public_evidence=public)

    def propose(self, observation, event=None):
        if event is None:
            raise ValueError("unscripted wakeups require an explicit closure event contract")
        if event.actor_id != self.profile.user_id:
            raise SimulatorInvalid("actor_session_mismatch", case_id=self.case.case_id,
                                   event_id=event.event_id)
        rule = getattr(self, "event_pacing_rules", {}).get(event.event_id)
        if rule is not None:
            # Only execution-time overlap permission changes. Preserve the
            # source case, exact words, identity, intent and scoring contract.
            event = event.model_copy(update={"overlap_allowed": rule.effective_overlap_allowed,
                                            "max_delay_ms": 0})
        enricher = getattr(self, "observation_enricher", None)
        if enricher is not None and not (event.fixed_utterance and event.fixed_utterance_exact):
            observation = enricher(observation)
        try:
            context = self._contract_context(observation, event)
        except (ValueError, TypeError) as exc:
            raise SimulatorInvalid("invalid_actor_observation_or_event", case_id=self.case.case_id,
                event_id=event.event_id, audit=[{"error": str(exc)}]) from exc
        if not self.runtime.present and event.action != UserAction.JOIN:
            raise SimulatorInvalid("absent_actor_cannot_speak", case_id=self.case.case_id,
                                   event_id=event.event_id)
        prompt = writer_prompt(context, alignment_ms=self.timing_alignment_ms)
        # Failed candidates are evidence, not durable user memories. Future
        # turns retain only accepted writer JSON, never critic answers.
        messages = list(self.messages) + [{"role": "user", "content": prompt}]
        event_audit = []
        for attempt in range(4):
            record = {"schema": CONTRACT_VERSION, "event_id": event.event_id,
                      "actor_id": event.actor_id, "attempt": attempt,
                      "logical_ms": observation.sim_time_ms, "accepted": False,
                      "context": context}
            event_audit.append(record)
            self.contract_audit.append(record)
            try:
                exact = bool(event.fixed_utterance and event.fixed_utterance_exact)
                if exact:
                    raw = {"actor_id": event.actor_id, "event_id": event.event_id,
                           "action": event.action.value, "utterance": event.fixed_utterance,
                           "addressee": event.addressee, "urgency": event.urgency,
                           "intent": event.intent, "overlap_allowed": event.overlap_allowed,
                           "thread_id": event.thread_id, "speech_act": event.speech_act,
                           "requested_delay_ms": 0, "target_start_ms": observation.sim_time_ms,
                           "max_wait_ms": event.max_delay_ms}
                else:
                    raw = self.backend.generate_json(messages)
                # The event scheduler, not the slow user-planning LLM, owns
                # the timestamp of a zero-delay event. During a paused
                # logical clock the LLM can return a stale target_start_ms
                # (or requested_delay_ms) even though the event is explicitly
                # constrained to start now. Normalize only timing fields;
                # actor, utterance, intent, thread and permissions remain
                # model-generated and are still checked below.
                if (not exact and isinstance(raw, dict)
                        and int(event.max_delay_ms or 0) == 0):
                    raw = dict(raw)
                    raw["requested_delay_ms"] = 0
                    raw["target_start_ms"] = int(observation.sim_time_ms)
                    raw["max_wait_ms"] = 0
                record["raw_candidate"] = raw
                candidate = validate_candidate(raw, case=self.case, profile=self.profile,
                    event=event, context=context, alignment_ms=self.timing_alignment_ms)
                if not exact:
                    verdict = critique_candidate(self.backend, context=context, candidate=candidate)
                    record["semantic_verdict"] = verdict.model_dump(mode="json")
                    if not verdict.accepted:
                        failed = [k for k, c in verdict.checks.items() if not c.passed]
                        raise ValueError("semantic user contract rejected: " + ", ".join(failed))
                else:
                    record["semantic_scope"] = "exact_authored_speech_static_contract_only"
                self._proposal_index += 1
                data = candidate.model_dump(exclude={"event_id", "grounding"})
                proposal = UserProposal(proposal_id=f"{self.profile.user_id}:proposal:{self._proposal_index}",
                    triggered_by_event_id=event.event_id, required_event=event.required, **data)
                record["accepted"] = True
                self.messages.extend([{"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)}])
                return proposal
            except (ValueError, TypeError) as exc:
                record["error"] = str(exc)
                record["exhausted"] = attempt == 4
                messages.extend([{"role": "assistant", "content": json.dumps(record.get("raw_candidate"), ensure_ascii=False)},
                    {"role": "user", "content": json.dumps({"repair": str(exc),
                        "semantic_checks": record.get("semantic_verdict"),
                        "instruction": "Generate a new USER utterance satisfying the same event and evidence. "
                            "Do not rename an actor or preserve an assistant answer as user speech. "
                            "Do not change facts, permissions or logical time; no fallback to reciting the intent."},
                        ensure_ascii=False)}])
        raise SimulatorInvalid("proposal_contract_exhausted", case_id=self.case.case_id,
                               event_id=event.event_id, audit=event_audit)


class OrderedOnlineEventGraph(EventGraphExecutor):
    """Preserve the old compiler's declaration order and completed-user prefix.

    Do not latch future e_live_followup on the first assistant speech start.
    The event definitions themselves are not edited.
    """

    def __init__(self, case, *, pacing_plan=None):
        super().__init__(case)
        self.pacing = None
        if pacing_plan is not None:
            from .pacing import PacingController
            self.pacing = PacingController(pacing_plan)
        self.pacing_audit = []
        self._last_pacing_reason = {}
        self._observed_current_onset = None
        self._unrealized_ids = set()

    def _observe_current_onset(self, observation):
        # Scheduled user audio must not blind the graph to real model onsets.
        # Record only already-played media; do not latch any future event here.
        if self.pacing is None or self.pacing.now_ms != observation.sim_time_ms:
            return
        start = self.pacing.current_speech_start_ms
        if start is None:
            self._observed_current_onset = None
        elif (self._observed_current_onset is None
              or self._observed_current_onset["start_ms"] != start):
            self._observed_current_onset = {
                "start_ms": start, "observed_at_ms": observation.sim_time_ms}

    def _matching_ordered_trigger_path(self, event, observation):
        if (self.pacing is not None
                and event.trigger.kind == TriggerKind.ASSISTANT_SPEECH_START
                and self._observed_current_onset is not None):
            # The onset is measured on played PCM, not the end of the first
            # native tick that reports it. This also catches a current reply
            # that began while the preceding user was still scheduled. Speech
            # is scheduled only NOW after all ordinary causal/prefix gates pass.
            onset = self._observed_current_onset["start_ms"]
            self._condition_seen_at_ms.setdefault(event.event_id, onset)
            self._condition_seen_kind.setdefault(event.event_id, ObservationKind.ASSISTANT_SPEECH_START)
        return self._matching_trigger_path(event, observation)

    def mark_completed(self, event_id, sim_time_ms):
        first_completion = event_id not in self._completed_at_ms
        super().mark_completed(event_id, sim_time_ms)
        if first_completion and self.pacing is not None:
            self.pacing.mark_event_completed(event_id, sim_time_ms)

    def register_event(self, event):
        is_new = event.event_id not in self._event_by_id
        super().register_event(event)
        if is_new and self.pacing is not None:
            self.pacing.register_event(event)


    def _presence_grant_targets(self, event):
        """Actors that would become present if this JOIN finished with presence_effects."""
        if event.action != UserAction.JOIN:
            return []
        targets = []
        for effect in (event.presence_effects or []):
            if bool(getattr(effect, 'present', False)):
                targets.append(effect.target_user_id)
        return targets

    def _cancel_join_unrealized_dependents(self, join_event, logical_ms, join_reason):
        """Fix-C6-A: when a presence-granting JOIN ends terminal_unrealized, cancel
        downstream speak/interrupt for that still-absent actor so they never hit
        absent_actor_cannot_speak. reason_code: join_unrealized_dependent_cancelled.
        Does not invent speech; session remains scoreable on remaining primary events.
        """
        targets = set(self._presence_grant_targets(join_event))
        if not targets:
            return []
        cancelled = []
        for dep in self._events:
            if dep.event_id == join_event.event_id:
                continue
            if dep.event_id in self._completed_at_ms or dep.event_id in self._unrealized_ids:
                continue
            if dep.event_id in self._scheduled_ids:
                # Already latched onto the audio path; do not silently drop mid-flight.
                continue
            if dep.actor_id not in targets:
                continue
            if dep.action not in (UserAction.SPEAK, UserAction.INTERRUPT):
                continue
            self._unrealized_ids.add(dep.event_id)
            self.pacing_audit.append({
                'event_id': dep.event_id,
                'logical_ms': logical_ms,
                'terminal_unrealized': True,
                'reason': 'join_unrealized_dependent_cancelled',
                'reason_code': 'join_unrealized_dependent_cancelled',
                'evidence': {
                    'cancelled_because_join_event_id': join_event.event_id,
                    'join_terminal_reason': join_reason,
                    'actor_id': dep.actor_id,
                    'action': dep.action.value if hasattr(dep.action, 'value') else str(dep.action),
                    'presence_still_false': True,
                    'fix': 'Fix-C6-A',
                },
                'scoring_policy': 'event_not_completed_but_session_remains_scoreable',
            })
            cancelled.append(dep.event_id)
        return cancelled

    def ready_events(self, observation):
        self._observe_current_onset(observation)
        for event in self._events:
            if event.event_id in self._completed_at_ms or event.event_id in self._unrealized_ids:
                continue
            if event.event_id in self._scheduled_ids:
                return []
            path = self._matching_ordered_trigger_path(event, observation)
            if self.pacing is not None:
                decision = self.pacing.gate(event.event_id, observation.sim_time_ms,
                    max(self._completed_at_ms.values(), default=0), path)
                decision.evidence.update({
                    "causal_onset_ms": self._condition_seen_at_ms.get(event.event_id),
                    "current_onset_observed_at_ms": (self._observed_current_onset or {}).get("observed_at_ms"),
                    "trigger_kind": event.trigger.kind.value})
                state = (decision.allow, decision.reason, decision.trigger_path)
                if state != self._last_pacing_reason.get(event.event_id):
                    self.pacing_audit.append({"event_id": event.event_id,
                        "logical_ms": observation.sim_time_ms,
                        "raw_trigger_path": path, "allow": decision.allow,
                        "reason": decision.reason, "trigger_path": decision.trigger_path,
                        "execution_max_delay_ms": 0,
                        "original_max_delay_ms": event.max_delay_ms,
                        "effective_overlap_allowed": self.pacing.rules[event.event_id].effective_overlap_allowed,
                        "evidence": decision.evidence})
                    self._last_pacing_reason[event.event_id] = state
                # Do not latch a rejected fallback: a later real assistant
                # onset must still be able to satisfy a critical interruption.
                if not decision.allow:
                    if decision.terminal:
                        self._unrealized_ids.add(event.event_id)
                        self.pacing_audit.append({"event_id": event.event_id,
                            "logical_ms": observation.sim_time_ms, "terminal_unrealized": True,
                            "reason": decision.reason, "evidence": decision.evidence,
                            "scoring_policy": "event_not_completed_but_session_remains_scoreable"})
                        # Fix-C6-A: join terminal_unrealized -> cancel still-absent
                        # dependents (speak/interrupt). reason_code:
                        # join_unrealized_dependent_cancelled. No Blake speech invented.
                        cancelled = self._cancel_join_unrealized_dependents(
                            event, observation.sim_time_ms, decision.reason)
                        if cancelled:
                            self.pacing_audit.append({
                                "event_id": event.event_id,
                                "logical_ms": observation.sim_time_ms,
                                "join_unrealized_dependents_cancelled": cancelled,
                                "reason_code": "join_unrealized_dependent_cancelled",
                                "fix": "Fix-C6-A",
                            })
                        return []
                    return []
                path = decision.trigger_path or path
            if path is not None:
                self._triggered_ids.add(event.event_id)
                self._trigger_paths.setdefault(event.event_id, path)
                self._trigger_observation_kinds.setdefault(event.event_id,
                    self._condition_seen_kind.get(event.event_id, observation.kind))
            return [event] if event.event_id in self._triggered_ids else []
        return []


class OnlineMultiUserSimulator(MultiUserVoiceSimulator):
    def __init__(self, **kwargs):
        pacing_plan = kwargs.pop("pacing_plan", None)
        if kwargs.get("frozen_plan") is not None:
            raise ValueError("online protocol forbids a frozen plan")
        super().__init__(**kwargs)
        if pacing_plan is None:
            from .pacing import build_case_pacing
            pacing_plan = build_case_pacing(self.case)
        self.event_graph = OrderedOnlineEventGraph(self.case, pacing_plan=pacing_plan)
        for session in self.sessions.values():
            session.event_pacing_rules = self.event_graph.pacing.rules

    def _wake_dynamic_users(self, observation, actors_with_events):
        # As with the previous compiler, only authored events and explicit
        # adaptive closure events can introduce speech. No unbounded free chat.
        return []

    def append_tail(self, event):
        from experiments.tau_multiuser_voice.event_actor_contract import validate_user_event_actor
        validate_user_event_actor(event.actor_id, case=self.case, event_id=event.event_id)
        compile_event_contract(self.case, event)
        self.event_graph.register_event(event)
        self._event_by_id[event.event_id] = event

    def _commit_finished_events(self, sim_time_ms):
        # ACL/presence effects are committed only after the corresponding user
        # PCM has finished. A late join never grants access to earlier speech.
        pending = [(end, self._event_by_id[event_id])
                   for event_id, end in self._event_end_ms.items()
                   if end <= sim_time_ms and event_id not in self._effects_applied]
        super()._commit_finished_events(sim_time_ms)
        for end, event in sorted(pending, key=lambda pair: pair[0]):
            for effect in event.presence_effects:
                session = self.sessions[effect.target_user_id]
                if hasattr(session, "record_presence"):
                    session.record_presence(effect.present, end)

    @property
    def all_events_completed(self):
        # Fix-C6-A / pacing scoring_policy: terminal_unrealized (incl. dependents
        # cancelled with join_unrealized_dependent_cancelled) are settled without
        # completion; remaining primary events can still close a scoreable session.
        settled = set(self.event_graph.completed_event_ids)
        settled |= getattr(self.event_graph, "_unrealized_ids", set())
        return set(self._event_by_id) <= settled

    def audio_integrity(self):
        """Validate the actual scheduled PCM, without inventing a frozen plan."""
        items = {item.metadata.get("event_id"): item
                 for queue in self.queues.values() for item in queue.items}
        rows = []
        for spoken in self.trace.spoken_events:
            item = items.get(spoken.event_id)
            if item is None or item.speaker_id != spoken.speaker_id:
                raise ValueError("online audio has no matching speaker/event PCM")
            if item.start_ms != spoken.start_ms or item.end_ms != spoken.end_ms:
                raise ValueError("online audio timing differs from its actual scheduled PCM")
            if not item.pcm_s16le or item.duration_ms <= 0:
                raise ValueError("online event audio is empty")
            import hashlib
            rows.append({"event_id": spoken.event_id, "speaker_id": spoken.speaker_id,
                "start_ms": item.start_ms, "end_ms": item.end_ms,
                "sample_rate": item.sample_rate, "samples": len(item.pcm_s16le) // 2,
                "pcm_sha256": hashlib.sha256(item.pcm_s16le).hexdigest(),
                "duration_repaired": False})
        if not rows:
            raise ValueError("no user events were played")
        return {"schema_version": "reviewed-online-audio-integrity-v1", "passed": True,
                "event_ids": [r["event_id"] for r in rows], "event_count": len(rows),
                "scope": "actual_scheduled_pcm_no_resynthesis_or_duration_repair", "events": rows}


class OnlineClosurePlanner(ModelTailEventOrchestrator):
    """Same tail contract/budget, decided from this session's played history."""

    def _validate_or_repair(self, *, messages, payload, case, round_index):
        from experiments.tau_multiuser_voice.event_actor_contract import (
            InvalidUserEventActor, validate_user_event_actor,
        )
        # Two semantic re-plans after the first invalid candidate. Every
        # candidate remains in the audit; never rename the assistant into a
        # participant or silently turn a required repair into a completed turn.
        candidate = payload
        repair_messages = list(messages)
        allowed = [user.user_id for user in case.users]
        for attempt in range(5):
            audit_start = len(self.audit_log)
            decision = None
            try:
                if candidate is None:
                    raise ValueError("closure planner returned no JSON object")
                decision = self._validate_schema_candidate(messages=repair_messages,
                    payload=candidate, case=case, round_index=round_index)
                if decision.action == "respond":
                    explicit = candidate.get("actor_id")
                    if explicit:
                        # Other alias fields cannot launder actor_id=assistant
                        # into a valid participant. Unique aliases of the
                        # explicit actor itself retain the old normalization.
                        explicit_actor = self._normalise_actor_id(
                            {"action": "respond", "actor_id": explicit}, case)["actor_id"]
                        validate_user_event_actor(explicit_actor, case=case,
                                                 event_id=f"tail_{round_index:02d}")
                    validate_user_event_actor(decision.actor_id, case=case,
                                             event_id=f"tail_{round_index:02d}")
                    compile_event_contract(case, self.to_event(decision, round_index,
                        getattr(self, "_closure_context", {}).get("logical_now_ms", 0)))
                if getattr(self, "_closure_context", None) is not None:
                    verdict = critique_closure(self.backend, context=self._closure_context,
                                               decision=decision)
                    for entry in self.audit_log[audit_start:]:
                        entry["semantic_user_contract"] = verdict.model_dump(mode="json")
                    if not verdict.accepted:
                        raise ValueError("closure user-intent contract rejected: " + ", ".join(
                            name for name, check in verdict.checks.items() if not check.passed))
                for entry in self.audit_log[audit_start:]:
                    entry["actor_contract_valid"] = True
                    entry["semantic_actor_replan_index"] = attempt
                return decision
            except (InvalidUserEventActor, ValueError, TypeError) as exc:
                if len(self.audit_log) == audit_start:
                    self.audit_log.append({"round_index": round_index, "raw_payload": candidate})
                for entry in self.audit_log[audit_start:]:
                    entry.update(accepted=False, actor_contract_valid=not isinstance(exc, InvalidUserEventActor),
                        semantic_actor_replan_index=attempt,
                        actor_validation_error=str(exc), allowed_user_ids=allowed,
                        actor_replan_exhausted=attempt == 3)
                if attempt == 3:
                    if getattr(self, "_closure_context", None) is not None:
                        raise SimulatorInvalid("closure_contract_exhausted", case_id=case.case_id,
                            event_id=f"tail_{round_index:02d}", audit=self.audit_log[audit_start:]) from exc
                    raise
                repair_messages.extend([
                    {"role": "assistant", "content": json.dumps(candidate, ensure_ascii=False)},
                    {"role": "user", "content": json.dumps({
                        "error": str(exc),
                        "allowed_user_ids": allowed,
                        "invalid_actor_id": decision.actor_id if decision else (
                            candidate.get("actor_id") if isinstance(candidate, dict) else None),
                        "repair_instruction": "Re-plan the closure decision from the SAME played evidence. "
                            "Do not replace the actor name while retaining an assistant's answer as user intent. "
                            "If the assistant still owes an answer, an actual user may briefly request it; "
                            "never fabricate that answer as user speech. Choose end only if no outstanding "
                            "question or repair remains, not merely to bypass this error. Preserve the "
                            "existing event contract, identities, facts and logical timestamp. Return JSON."},
                        ensure_ascii=False)}])
                candidate = self.backend.generate_json(repair_messages)

    def _validate_schema_candidate(self, *, messages, payload, case, round_index):
        # An explicit end creates no user event. Null response-only fields mean
        # the same thing as omitting those fields; use the existing schema
        # defaults, without changing action, actor, reason, or any respond event.
        # Keep the unmodified candidate in the audit rather than concealing this
        # formatting normalization. Actor semantic re-planning is bounded in
        # the caller rather than hidden inside this schema-only normalization.
        if not isinstance(payload, dict):
            raise ValueError("closure candidate must be a JSON object")
        original = dict(payload)
        if getattr(self, "_closure_context", None) is not None:
            aliases = {"user_id", "user", "participant_id", "actor", "speaker", "selected_user",
                       "selected_participant", "participant_name", "name", "requester", "participant"}
            unknown = set(payload) - set(TailDecision.model_fields) - aliases
            if unknown:
                raise ValueError("closure candidate invented fields: " + ", ".join(sorted(unknown)))
            if "action" not in payload or not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
                raise ValueError("closure requires explicit action and nonempty reason; missing JSON is not end")
        normalized = dict(payload)
        replaced = {}
        if payload.get("action") == "end" and payload.get("actor_id") is None:
            for name in ("intent", "addressee", "thread_id", "speech_act", "urgency",
                         "overlap_allowed", "max_delay_ms", "requested_delay_ms"):
                if name in normalized and normalized[name] is None:
                    default = TailDecision.model_fields[name].default
                    normalized[name] = default
                    replaced[name] = {"from": None, "to": default}
        audit_start = len(self.audit_log)
        try:
            if getattr(self, "_closure_context", None) is not None:
                # The online path has one shared bound of three raw candidates.
                # Do not let the legacy inner formatting retry introduce an
                # unreviewed second JSON or silently ignore its extra fields.
                normalized = self._normalise_actor_id(normalized, case)
                entry = {"round_index": round_index, "raw_payload": original,
                         "normalized_payload": dict(normalized),
                         "repair_attempted": False, "accepted": False}
                self.audit_log.append(entry)
                decision = TailDecision.model_validate(normalized, strict=True)
                entry["accepted"] = True
                return decision
            return super()._validate_or_repair(messages=messages, payload=normalized,
                                              case=case, round_index=round_index)
        finally:
            for entry in self.audit_log[audit_start:]:
                entry["raw_payload"] = original
                if replaced:
                    entry["online_end_null_normalization"] = replaced

    def decide_online(self, *, case, observation, spoken_events, round_index,
                      played_timeline=None, completion_context=None):
        causal_observation_payload(observation)
        context = closure_public_context(case, observation, played_timeline)
        participants = [{"user_id": u.user_id, "display_name": u.display_name,
                         "role": u.role} for u in case.users]
        messages = [{"role": "system", "content":
            "You manage closure of one ongoing multi-user conversation. Use ONLY supplied played public content. "
            "Choose action=end only when both sides have finished the exchange: no user or assistant "
            "question, requested confirmation, promised result, or misunderstanding remains unanswered. "
            "In particular, the final user asking 'Is that correct?' is NOT a completed conversation "
            "until the assistant has had an opportunity to answer it. Do not silently ignore pending user questions. "
            "Use the timestamped played_timeline and completion_context to distinguish an old assistant "
            "answer from a response AFTER the latest user request. A user's own recap or 'thank you for confirming' "
            "is not evidence that the assistant actually supplied the requested answer. Never credit user speech "
            "as assistant task completion. ASR interval bounds are not exact word alignments. "
            "Choose action=respond only to resolve an explicit outstanding question/repair; do not invent new tasks. "
            "If the assistant needs to finish an unanswered user request, a respond intent may ask the "
            "same user to briefly request that missing answer; never supply the answer on behalf of the assistant. "
            "Return JSON action, actor_id, intent, addressee, thread_id, speech_act, urgency (integer 0..5), "
            "overlap_allowed (boolean), max_delay_ms, requested_delay_ms, reason. Respond requires a listed canonical "
            "actor_id and a semantic intent, not a full scripted dialogue; end requires actor_id=null. "
            "The assistant is the evaluated model, NOT an allowed actor_id for user simulation. "
            "For end, omit response-only fields rather than assigning them null; reason must be a string. "
            "The supplied output_contract is mandatory. Pick speech_act ONLY from its allowed_speech_acts. "
            "Use the listed INTERNAL user_id values for actor_id and participant addressee. "
            "For speech to the assistant use addressee=assistant, not the assistant name; "
            "for the whole group use group. Do not use public p01/p02 codes or names in these "
            "simulator metadata fields. These fields are not spoken or shown to the tested model."},
            {"role": "user", "content": json.dumps({"round_index": round_index,
                "allowed_user_actor_ids": [u.user_id for u in case.users],
                "output_contract": {
                    "allowed_speech_acts": sorted(__import__(
                        'experiments.tau_multiuser_voice.simulation_contract',
                        fromlist=['KNOWN_SPEECH_ACTS']).KNOWN_SPEECH_ACTS),
                    "allowed_actor_ids": [u.user_id for u in case.users],
                    "allowed_addressees": [u.user_id for u in case.users] + ['assistant', 'group', None],
                    "namespace": "simulator_internal_ids_not_model_tool_public_codes",
                    "response_schema": TailDecision.model_json_schema(),
                },
                "participants": participants,
                "observation": {"sim_time_ms": observation.sim_time_ms,
                                "assistant_speaking": observation.assistant_speaking},
                "played_timeline": context["played_evidence"],
                "completion_context": completion_context or {},
                "played_user_events": [{"speaker": e.speaker_id, "text": e.utterance,
                    "start_ms": e.start_ms, "end_ms": e.end_ms} for e in spoken_events
                    if e.end_ms <= observation.sim_time_ms]}, ensure_ascii=False)}]
        payload = self.backend.generate_json(messages)
        self._closure_context = context
        try:
            return self._validate_or_repair(messages=messages, payload=payload,
                                            case=case, round_index=round_index)
        finally:
            del self._closure_context

    @staticmethod
    def to_event(decision, round_index, now_ms):
        return ScriptedEvent(event_id=f"tail_{round_index:02d}", actor_id=decision.actor_id,
            action=UserAction.SPEAK, trigger=TriggerSpec(kind=TriggerKind.ELAPSED, after_ms=now_ms),
            intent=decision.intent, addressee=decision.addressee,
            thread_id=decision.thread_id, speech_act=decision.speech_act,
            urgency=decision.urgency, overlap_allowed=decision.overlap_allowed,
            max_delay_ms=decision.max_delay_ms, required=True,
            notes=["adaptive closure in the same persistent model session"])
