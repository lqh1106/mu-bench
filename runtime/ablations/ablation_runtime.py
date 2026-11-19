"""Hooks imported only by the generated, private ablation runner."""
from dataclasses import replace
import json
from pathlib import Path
from ablation_core import Experiment, VERSION, write


def prepare(case, args):
    arm=args.ablation_arm
    if not getattr(args,'openweight_endpoint',None):
        raise ValueError('provider endpoint required')
    if case.expected_tool_actions:
        raise ValueError('Audio ablations are restricted to tool-free cases')
    experiment=Experiment(arm,case.model_dump(mode='json'),args.output_dir/case.case_id)
    from ablation_version import scoring_version
    runtime=Path(__import__('os').environ['MUBENCH_RUNTIME_ROOT'])
    experiment.manifest.update(scoring_version(runtime))
    import os
    compiled=Path(os.environ['MUBENCH_ABLATION_CASE_DIR'])/arm/case.case_id
    saved=json.loads((compiled/'manifest.json').read_text())
    for key in ('source_case_sha256','transformed_case_sha256','task_contract_sha256','task_success_policy'):
        if saved[key]!=experiment.manifest[key]:
            raise ValueError('compiled ablation version mismatch: '+key)
    if json.loads((compiled/'case.json').read_text()) != experiment.transformed:
        raise ValueError('compiled ablation case differs from runtime transform')
    from experiments.mu_bench.logical_time.pacing import URGENT_EVENT_EXEMPTIONS
    for event in case.events:
        if (case.variant.scenario_id, event.event_id) in URGENT_EVENT_EXEMPTIONS:
            experiment.manifest['paired_review_reasons'].append(event.event_id + ':urgent_preemptive_intervention')
    experiment.manifest['paired_review_reasons'] = sorted(set(experiment.manifest['paired_review_reasons']))
    # Same event action, actors, words, ACL, tools, state and goals. Only timing changes.
    changed=type(case).model_validate(experiment.transformed)
    return changed,experiment


def pacing(case, experiment):
    from experiments.mu_bench.logical_time.pacing import build_case_pacing
    plan=build_case_pacing(case)
    if not experiment.serial:return plan
    rules={}
    for i,event in enumerate(case.events):
        rule=plan.events[event.event_id]
        rules[event.event_id]=replace(rule,mode='response_opportunity',
            protected_causal=False,require_current_overlap=False,urgent_exemption=False,
            effective_overlap_allowed=False,min_current_response_played_ms=0,
            reasons=('Ablation: sequential audio turns after response/playback drain.',))
    return replace(plan,events=rules,version=VERSION)


def user_session_class(experiment):
    from experiments.mu_bench.logical_time.online_users import OnlineUserSession
    if not experiment.semantics:return OnlineUserSession
    class SemanticInputSession(OnlineUserSession):
        def propose(self,observation,event=None):
            proposal=super().propose(observation,event)
            experiment.after_proposal(self,event,proposal)
            return proposal
    return SemanticInputSession


def adapt_prompt(prompt,tools,experiment):
    if not prompt and not tools:return prompt,tools  # MOSS uses acknowledged auxiliary records.
    if not experiment.identity:return prompt,tools
    # This protocol has automatic, ground-truth requester binding. Leaving the
    # old mandatory attribution tool would reintroduce the ablated ability.
    start=prompt.index('Attribution protocol (required because the benchmark registers this function):')
    boundaries=[prompt.find(x,start) for x in ('\nTool protocol:', '\n\nInformation available to the assistant before this conversation')]
    stop=min((x for x in boundaries if x>=0),default=len(prompt))
    prompt=prompt[:start]+'''Trusted routing protocol:
Each user audio turn has a machine-provided routing record: speaker, addressee,
request owner and thread. These labels are authoritative. They do not authorize
an action. The runner binds business calls to the actual completed speaker;
continue to enforce permissions, consent, privacy and all business constraints.
Do not call report_speaker_attribution; it is not registered in this condition.
'''+prompt[stop:]
    prompt=prompt.replace('Establish the speaker and addressee through the attribution call before selecting a tool.',
                          'Use the trusted routing record when selecting a tool.')
    prompt=prompt.replace('The runner binds the trusted requester from the attribution report.',
                          'The runner binds the trusted requester from the completed user audio turn.')
    if experiment.semantics:
        prompt+='\nA structured_user_input record accompanies the completed audio. It is a faithful parse of that user input, not a solution or permission grant. Plan and execute the requested task, resolve conflicting constraints, and verify tool results yourself.\n'
    return prompt,[t for t in tools if t.openai_schema['function']['name']!='report_speaker_attribution']


def automatic_binding(experiment,adapter,trace,window,in_task,task_ms,session_ms,attributed,enrollment):
    if not experiment.serial:return None
    event=experiment.prepare_tick(adapter,trace,window.active_speakers,in_task,task_ms,session_ms,enrollment)
    if event is None:return None
    from experiments.tau_multiuser_voice.models import SpeakerAttributionTrace
    trace.speaker_attributions.append(SpeakerAttributionTrace(timestamp_ms=session_ms,
        task_timestamp_ms=task_ms,predicted_speaker_id=event.speaker_id,
        predicted_addressee=event.addressee or 'unknown',predicted_thread_id=event.thread_id,
        confidence=1.0,reported_speaker_reference=event.speaker_id,
        reported_addressee_reference=event.addressee or 'unknown',
        matched_event_id=event.event_id,matched_proposal_id=event.proposal_id))
    attributed.add(event.proposal_id)
    experiment.context_audit.append(dict(kind='oracle_requester_binding',event_id=event.event_id,
        actor=event.speaker_id,logical_ms=session_ms,source='ground_truth_completed_user_audio',
        is_model_prediction=False))
    experiment.save()
    return event.speaker_id


def summarize(experiment,output_dir):
    """Sidecar only: never replace the task scorer or change its score.json."""
    experiment.save()
    p=Path(output_dir)/'score.json'
    if p.exists():
        score=json.loads(p.read_text())
        write(Path(output_dir)/'ablation_result.json',dict(schema=VERSION,arm=experiment.arm,
            case_id=experiment.source['case_id'],source_case_sha256=experiment.manifest['source_case_sha256'],
            task_contract_sha256=experiment.manifest['task_contract_sha256'],
            scoring_sha256=experiment.manifest['scoring_sha256'],
            task_success_policy=experiment.manifest['task_success_policy'],
            task_success=score['task_success'],score_is_unmodified=True,
            paired_review_reasons=experiment.manifest['paired_review_reasons'],
            scope='task_success_only; oracle routing is not measured recognition performance'))
