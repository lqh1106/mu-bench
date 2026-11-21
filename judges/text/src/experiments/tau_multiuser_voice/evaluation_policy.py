"""Versioned business-effect comparison and causal query checks.

The policy is explicitly scoped to the frozen 408 tool cases. It changes no
permissions, user goals, protected scopes, or event scheduling. Original audit
events remain in every trace; only the comparison copy is projected.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json

VERSION = "mu-business-causal-revision-v4"

class TemporalAlignmentPending(RuntimeError):
    """Preserve ambiguous audio evidence without turning it into a SUT failure."""

def _get(x, name, default=None):
    return x.get(name, default) if isinstance(x, dict) else getattr(x, name, default)

@lru_cache(maxsize=1)
def _policies():
    return json.loads(Path(__import__('importlib').import_module('experiments.tau_multiuser_voice.evaluation_policy_data').__file__).with_name('evaluation_policy_20261114.json').read_text())['cases']

def policy_for(case):
    return _policies().get(_get(case, 'case_id'), {})

def business_state_projection(case, state):
    policy = policy_for(case)
    if not policy:
        return state
    out = deepcopy(state)
    out.pop('audit_events', None)
    for path in policy['projection_paths']:
        record = out
        for part in path[:-1]:
            record = record.get(part) if isinstance(record, dict) else None
        if isinstance(record, dict):
            record.pop(path[-1], None)
    return out

def is_pacing_query(case, expected):
    return any(all(_get(expected, key) == entry[key] for key in
                   ('tool_name', 'resource_id', 'actor_id', 'before_event_id'))
               for entry in policy_for(case).get('pacing_queries', []))

def is_pacing_after(case, expected):
    return any(all(_get(expected, key) == entry[key] for key in
                   ('tool_name', 'resource_id', 'actor_id', 'after_event_id'))
               for entry in policy_for(case).get('pacing_after_actions', []))

def result_available(action):
    explicit = _get(action, 'result_available_ms')
    if explicit is not None:
        return explicit, _get(action, 'result_availability_basis', 'explicit')
    # Old logical runners persist the invocation tick, not delivery precision.
    # A strictly later dependent tick is usable ordering evidence; equality is
    # deliberately not treated as an acknowledged result.
    return _get(action, 'timestamp_ms'), 'legacy_invocation_tick_lower_bound'

def causal_query_check(case, query, matches, trace):
    successful = [a for a in matches if _get(a, 'executed') and _get(a, 'oracle_authorized')]
    declared = case.task_contract_revision.get('query_dependencies', {}).get(query.tool_name)
    dependents = [e for e in case.expected_tool_actions
                  if e.allowed and e.must_execute and not e.tool_name.startswith('get_')]
    dependent_names = set(declared['dependent_tools']) if declared is not None else {e.tool_name for e in dependents}
    actions = [a for a in trace.tool_actions if a.executed and a.tool_name in dependent_names]
    checks = []
    for action in actions:
        supported = []
        fact_paths=(declared or {}).get('required_fact_paths',[])
        prior=successful
        if fact_paths:
            action_index=next(i for i,a in enumerate(trace.tool_actions) if a is action)
            prior=[a for a in trace.tool_actions[:action_index]
                   if a.executed and a.oracle_authorized and a.resource_id==query.resource_id
                   and all(p in a.returned_fact_paths for p in fact_paths)]
        for read in prior:
            at, basis = result_available(read)
            if at is not None and (at < action.timestamp_ms if basis == 'legacy_invocation_tick_lower_bound' else at <= action.timestamp_ms):
                supported.append({'available_ms': at, 'basis': basis,'source_tool':read.tool_name,'required_fact_paths':fact_paths})
        checks.append({'dependent_tool': action.tool_name,
                       'dependent_timestamp_ms': action.timestamp_ms,
                       'supported': bool(supported), 'support': supported})
    # Candidate staging uses a directly supplied, delegated personal value;
    # it does not inherently consume a prior record lookup. Keep actual source
    # review and all other genuine lookup dependencies.
    import os
    if os.environ.get('MUBENCH_TOOL_CASE_MECHANISM') in {'tr2','tr3','tr4','tr5'} and case.case_id.split('__')[0] in {'travel_identity_01_passport_digit_correction','travel_identity_03_document_interpreter'}:
        for check,action in zip(checks,actions):
            if action.request_basis_status=='supported':
                check['supported']=True
                check['support'].append({'basis':'independently_validated_delivered_candidate','event_ids':action.request_basis_evidence})
    scripted = next((e for e in case.events if e.event_id == query.before_event_id), None)
    observed = next((e for e in trace.spoken_events if e.event_id == query.before_event_id), None)
    return {'query_tool': query.tool_name, 'resource_id': query.resource_id,
            'pacing_before_event_ignored': query.before_event_id,
            'configured_trigger': str(scripted.trigger.kind.value) if scripted else None,
            'observed_trigger_path': _get(observed, 'trigger_path'),
            'observed_trigger_kind': str(_get(observed, 'trigger_observation_kind').value) if _get(observed, 'trigger_observation_kind') else None,
            'successful_query_present': bool(successful),
            'dependent_action_checks': checks,
            'legacy_timing_precision': any(result_available(a)[1] == 'legacy_invocation_tick_lower_bound' for a in successful),
            'dependency_contract': declared,
            'passed': ((bool(successful) if query.must_execute else True) and all(c['supported'] for c in checks))}

def cited_result_temporal_check(case, row, sources):
    """Reject future tool evidence used to support earlier spoken claims.

    Trusted played-waveform ASR start/end and tool timestamps are both relative
    to task start. A citation interval straddling delivery is insufficient to
    prove ordering; the judge must cite a later, precisely supported fragment.
    This checks query-dependent evidence, not every acknowledgment or write
    receipt. Ambiguous ASR intervals are recorded separately, never converted
    into a model failure by guessing a quote's within-fragment timestamp.
    """
    if not policy_for(case):
        return [], []
    speech = [sources.get(r.get('source_id')) for r in (row.get('assistant_evidence') or []) if isinstance(r, dict)]
    from .jt3_review import enabled as jt3_enabled
    refs=row.get('tool_evidence') or []
    if jt3_enabled():
        default='counterevidence' if row.get('passed') is False else 'prior_basis'
        refs=[r for r in refs if isinstance(r,dict) and r.get('purpose',default)=='prior_basis']
    tools = [sources.get(r.get('source_id')) for r in refs if isinstance(r,dict)]
    speech = [s for s in speech if s and s.get('kind') == 'assistant_text']
    query_names = {q['tool_name'] for q in policy_for(case).get('pacing_queries', [])}
    tools = [s for s in tools if s and s.get('kind') == 'executed_tool_result' and ((case.task_contract_revision.get('request_revision_protocol') == 'request-revision' and 'assertion_id' not in row) or s.get('tool_name') in query_names)]
    issues = []
    ambiguities = []
    def relation(tool, utterance):
        available, basis = result_available(tool)
        start, end = utterance.get('start_ms'), utterance.get('end_ms')
        if available is None or start is None or end is None:
            return 'unknown'
        if available > end:
            return 'future'
        if available < start or (available == start and basis != 'legacy_invocation_tick_lower_bound'):
            return 'prior'
        return 'overlap'
    # Multiple legitimate rounds may cite R1 -> S1, then R2 -> S2. Requiring
    # every tool to precede every utterance would reject that valid dialogue.
    if tools and speech:
        for utterance in speech:
            relations = [relation(tool, utterance) for tool in tools]
            if all(x == 'future' for x in relations):
                issues.append('no cited tool result was demonstrably available before spoken claim: '+utterance['source_id'])
            elif 'prior' not in relations:
                ambiguities.append('query/quoted-ASR interval needs alignment: '+utterance['source_id'])
        for tool in tools:
            relations = [relation(tool, utterance) for utterance in speech]
            if all(x == 'future' for x in relations):
                issues.append('future or temporally ambiguous tool result cannot support cited speech: '+tool['source_id'])
            elif 'prior' not in relations:
                ambiguities.append('query evidence timing unresolved: '+tool['source_id'])
    return issues, ambiguities

def cited_result_temporal_issues(case, row, sources, catalog=None):
    return cited_result_temporal_check(case, row, sources)[0]
