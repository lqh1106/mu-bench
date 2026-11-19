"""Isolated, nested audio capability ablations. No provider/API imports."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = 'mu-nt-r3-ablation-participants-v3-20261117'
ARMS = ('baseline', 'serial', 'serial_identity', 'serial_identity_semantics')
TASK_FIELDS = ('users', 'goal_relations', 'expected_tool_actions', 'state_assertions',
               'final_success_conditions', 'environment', 'seed', 'task_contract_revision')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def task_digest(case):
    return digest({k: case.get(k) for k in TASK_FIELDS})


def transform_case(source, arm):
    """Keep IDs, speech meaning, users, ACL and task oracle; change only release timing."""
    if arm not in ARMS:
        raise ValueError('unknown ablation arm')
    result = deepcopy(source)
    changes = []
    review_reasons = []
    from_types = {'assistant_speech_start', 'assistant_prefix', 'tool_call', 'tool_result'}
    previous = None
    for event in result['events']:
        original = deepcopy(event)
        trigger = event['trigger']
        # Outcome-sensitive cases need a human semantic match review before inclusion
        # in paired causal estimates; they are not silently counted as easier cases.
        if trigger['kind'] in {'tool_call', 'tool_result'}:
            review_reasons.append(event['event_id'] + ':tool_transition_dependency')
        if event['action'] == 'revoke' or event.get('speech_act') == 'cancel_or_revoke':
            review_reasons.append(event['event_id'] + ':revocation_timing')
        if arm != 'baseline':
            event['trigger'] = dict(kind='conversation_start' if previous is None else 'event_completed',
                pattern=None, event_id=previous, tool_name=None, after_ms=0,
                fallback_at_ms=None, fallback_after_event_id=None)
            event['overlap_allowed'] = False
            event['max_delay_ms'] = 0
            # No rewrite of the authored words, intentions or speech act. Interrupt
            # proposals are scheduled as non-overlapping corrections, not deleted.
            changes.append(dict(event_id=event['event_id'], original=original,
                                transformed=deepcopy(event)))
        previous = event['event_id']
    assert task_digest(source) == task_digest(result)
    manifest = dict(schema=VERSION, arm=arm, case_id=source['case_id'],
        source_case_sha256=digest(source), transformed_case_sha256=digest(result),
        task_contract_sha256=task_digest(source), task_contract_unchanged=True,
        event_changes=changes, paired_review_reasons=sorted(set(review_reasons)),
        primary_metric='task_success',
        non_comparable_metrics=[] if arm=='baseline' else
            ['timing', 'floor_control', 'interruption_recovery', 'strict_joint_success'],
        identity_metrics_are_oracle_assisted=ARMS.index(arm)>=2,
        semantic_input_is_oracle_assisted=ARMS.index(arm)>=3,
        user_audio_preserved=True, future_event_information_released=False)
    return result, manifest


def routing_payload(case, event):
    users = {u['user_id']: u for u in case['users']}
    public = {uid: f'p{i+1:02d}' for i, uid in enumerate(users)}
    uid = event['speaker_id']
    user = users[uid]
    addressee = event.get('addressee')
    if addressee in public:
        addressee = public[addressee]
    elif addressee == 'assistant':
        addressee = case['assistant_activation']['assistant_name']
    thread_users = {'primary': list(users)[0], 'secondary': list(users)[1]}
    thread_users.update({f'participant_{i+1}': uid for i,uid in enumerate(users)})
    owner = event['thread_id'] if event['thread_id'] in users else thread_users.get(event['thread_id'])
    return dict(schema=VERSION, kind='trusted_routing',
        event_id=event['event_id'], request_id=event['event_id'],
        speaker_id=public[uid], speaker_name=user['display_name'], speaker_role=user['role'],
        addressee=addressee, requester_id=public[uid],
        request_owner_id=public.get(owner),
        thread_id=public.get(owner, event['thread_id']))


ORACLE_KEYS = {'goal', 'operation', 'arguments', 'constraints'}
OPERATIONS = {'request', 'question', 'correction', 'confirmation', 'backchannel',
              'side_talk', 'consent', 'revocation', 'join', 'leave', 'clarification_request', 'other'}


def validate_semantics(data, utterance, history=()):
    """Ground every semantic field in this actual utterance, never judge state."""
    if not isinstance(data, dict) or set(data) != ORACLE_KEYS:
        raise ValueError('semantic oracle schema mismatch')
    if data['operation'] not in OPERATIONS:
        raise ValueError('invalid semantic operation')
    if not isinstance(data['goal'], dict) or set(data['goal']) != {'value', 'evidence'}:
        raise ValueError('goal needs value and source quote')
    evidence_texts = [utterance] + [r["text"] for r in history]
    fields = [data['goal']]
    for key in ('arguments', 'constraints'):
        if not isinstance(data[key], list): raise ValueError('expected semantic list')
        for row in data[key]:
            expected = {'name', 'value', 'evidence'} if key=='arguments' else {'value', 'evidence'}
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError('invalid semantic field')
            fields.append(row)
    for row in fields:
        if not isinstance(row['value'], (str, int, float, bool)) or row['value'] == '':
            raise ValueError('empty semantic value')
        if not isinstance(row['evidence'], str) or not row['evidence'].strip() or not any(row['evidence'] in text for text in evidence_texts):
            raise ValueError('oracle fact has no exact spoken evidence')
    return deepcopy(data)


SEMANTIC_RETRY_POLICY = 'feedback-three-attempts-final-adopt-v1'

def compile_semantics(backend, utterance, routing, audit, history=()):
    """Feed validation detail into retries; adopt the third candidate with an explicit audit flag."""
    request = dict(task='Represent this user utterance as structured INPUT to an assistant. Do not solve it.',
        utterance=utterance, routing=routing, already_heard_history=list(history),
        schema={'goal': {'value':'user requested goal, not its solution','evidence':'exact quote'},
                'operation':'one string selected from allowed_operations',
                'arguments':[{'name':'slot name','value':'spoken value','evidence':'exact quote'}],
                'constraints':[{'value':'spoken constraint','evidence':'exact quote'}]},
        allowed_operations=sorted(OPERATIONS),
        rules=['Return exactly four top-level keys: goal, operation, arguments, constraints.',
               'operation is one string from allowed_operations, never a list. All evidence must be an exact substring of the utterance or already heard history; do not insert ellipses or normalize punctuation.',
               'Every value must be nonempty. For requested but unknown facts, describe what is requested; never invent an answer or use an empty value.',
               'Include all stated goals and constraints, including negations.',
               'Preserve corrections, uncertainty and non-directive speech.',
               'Resolve references using already_heard_history; current corrections override prior statements.',
               'No inferred tool results, authorization decision, hidden facts or proposed solution.',
               'Do not turn a backchannel or peer-directed statement into an assistant command.'])
    messages=[dict(role='system',content='Return only the requested JSON. You are a faithful input parser, not the task agent.'),
              dict(role='user',content=json.dumps(request,ensure_ascii=False))]
    for attempt in range(3):
        row=dict(attempt=attempt+1, utterance=utterance, routing=routing,
                 already_heard_history=list(history), retry_policy=SEMANTIC_RETRY_POLICY)
        audit.append(row)
        raw=None
        try:
            raw=backend.generate_json(messages);row['candidate']=deepcopy(raw)
            parsed=validate_semantics(raw,utterance,history)
            row['syntax_and_quotes_valid']=True
            if attempt==2:
                row.update(accepted=True, validation_passed=None, independent_review_passed=None,
                           forced_accept=True, adoption_reason='third_candidate_directly_adopted', review_skipped=True)
                return parsed
            check=backend.generate_json([
                dict(role='system',content='Check a structured user-input parse. Return JSON {"faithful":bool,"complete":bool,"no_solution_or_extra_facts":bool,"reason":str}. Reject changed intent, missing constraints or invented answers.'),
                dict(role='user',content=json.dumps(dict(utterance=utterance,already_heard_history=list(history),routing=routing,parse=parsed),ensure_ascii=False))])
            row['check']=check
            if not isinstance(check,dict) or any(check.get(k) is not True for k in ('faithful','complete','no_solution_or_extra_facts')):
                raise ValueError('semantic oracle review rejected')
            row.update(accepted=True,validation_passed=True,independent_review_passed=True,forced_accept=False)
            return parsed
        except (ValueError,TypeError) as exc:
            row.update(accepted=False,error=str(exc),validation_passed=False)
            if attempt==2:
                # A parse error does not invent a task answer. Preserve any JSON
                # candidate; if there is none, relay the actual utterance verbatim.
                candidate=deepcopy(raw) if isinstance(raw,dict) else dict(
                    goal=dict(value=utterance,evidence=utterance),operation='other',arguments=[],constraints=[])
                row.update(accepted=True,forced_accept=True,independent_review_passed=None,
                           review_skipped=True,adoption_reason='third_candidate_adopted_despite_validation_error',
                           verbatim_fallback=not isinstance(raw,dict),adopted_candidate=deepcopy(candidate))
                return candidate
            feedback=dict(error=str(exc), previous_candidate=raw, validation=row.get('check'),
                          utterance=utterance, allowed_operations=sorted(OPERATIONS),
                          instruction='Correct the rejected candidate. Address each failed check and its stated reason. Preserve the input meaning; use exact contiguous quotes. Do not solve the task.')
            row['retry_feedback']=feedback
            if raw is not None:messages.append(dict(role='assistant',content=json.dumps(raw,ensure_ascii=False)))
            messages.append(dict(role='user',content=json.dumps(feedback,ensure_ascii=False)))
    raise AssertionError('unreachable bounded semantic retry')


class Experiment:
    def __init__(self, arm, source, output):
        self.arm=arm; self.source=source; self.output=Path(output)
        self.serial=arm!='baseline'; self.identity=ARMS.index(arm)>=2; self.semantics=ARMS.index(arm)>=3
        self.semantic_updates={};self.oracle_audit=[];self.context_audit=[]
        self.last_active=None;self.bound_proposal=None
        transformed,self.manifest=transform_case(source,arm)
        self.transformed=transformed

    def save(self):
        write(self.output/'ablation_manifest.json',self.manifest)
        write(self.output/'ablation_original_case.json',self.source)
        write(self.output/'ablation_context_audit.json',self.context_audit)
        write(self.output/'ablation_semantic_oracle_audit.json',self.oracle_audit)

    def after_proposal(self, session, event, proposal):
        if not self.semantics or not proposal.utterance: return
        speaker=dict(event_id=event.event_id,speaker_id=event.actor_id,
                     addressee=event.addressee,thread_id=event.thread_id)
        routing=routing_payload(self.source,speaker)
        accepted = next(r for r in reversed(session.contract_audit)
                        if r.get('event_id') == event.event_id and r.get('accepted'))
        history = [{k:r[k] for k in ('speaker','text','start_ms','end_ms')}
                   for r in accepted['context']['allowed_evidence']
                   if r.get('source') == 'played_public_speech']
        public = {u['user_id']: f'p{i+1:02d}' for i,u in enumerate(self.source['users'])}
        history = [dict(row, speaker=public.get(row['speaker'], row['speaker'])) for row in history]
        try:
            parsed=compile_semantics(session.backend,proposal.utterance,routing,self.oracle_audit,history)
        finally:
            self.save()
        self.semantic_updates[event.event_id]=dict(schema=VERSION,kind='structured_user_input',
            event_id=event.event_id, routing=routing, transcript=proposal.utterance,
            task=parsed,
            source=('third_candidate_adopted_without_review' if self.oracle_audit[-1].get('forced_accept') else 'validated_parse_of_current_speech_and_already_heard_history'),
            authoritative_task_answer=False)
        self.save()

    def prepare_tick(self, adapter, trace, active_speakers, in_task, task_ms, session_ms, enrollment=None):
        active=[e for e in trace.spoken_events if e.start_ms<=task_ms<e.end_ms] if in_task else []
        if len(active)>1 or len(active_speakers)>1:
            raise ValueError('ablation_input_overlap_detected')
        current=active[0].event_id if active else None
        start=None;end=None
        if self.identity and current and current!=self.last_active:
            start=routing_payload(self.source,active[0].model_dump(mode='json'))
            self.context_audit.append(dict(phase='utterance_start',logical_ms=session_ms,payload=start))
        if self.semantics and self.last_active and current!=self.last_active:
            end=self.semantic_updates[self.last_active]
            self.context_audit.append(dict(phase='utterance_end',logical_ms=session_ms,payload=end))
        self.last_active=current
        input_active = bool(active)
        if not in_task and enrollment is not None:
            spans = list(enrollment.traces) + ([enrollment.activation_trace] if enrollment.activation_trace else [])
            input_active = any(e.start_ms <= session_ms < e.end_ms for e in spans)
        adapter.set_ablation_input(input_active,start,end)
        if start or end:self.save()
        completed=[e for e in trace.spoken_events if e.end_ms<=task_ms] if in_task else []
        if self.identity and completed and not active:
            latest=max(completed,key=lambda e:e.end_ms)
            if latest.proposal_id!=self.bound_proposal:
                self.bound_proposal=latest.proposal_id
                return latest
        return None
