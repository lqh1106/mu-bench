"""Medical-note source text: semantic prose, separate from document IDs.

Only the source-consent fixture opts in. Raw arguments, results and audit logs
are never rewritten by these comparison helpers.
"""
from copy import deepcopy
SCENE='healthcare_care_02_medication_list_source'
REVISION='medication-source-text-v2'
TEXT_RULE=(
    'source is free-text provenance, not a hidden code. Check its material claims '
    'against the actual delivered list evidence: who supplied it, which document, '
    'and whether patient confirmation occurred. Faithful descriptions need not '
    'equal written_list. A contradiction such as patient-confirmed dose, a wrong '
    'source person/document, or invented confirmation must fail the relevant '
    'execution/provenance completion checks even when structured fields are correct. '
    'Do not require wording that the public interface did not prescribe.'
    ' For provenance-description checks, cite actual delivered source utterances and '
    'the executed note field with tool purpose execution_effect; no assistant '
    'speech is needed to prove the content of an executed source parameter. '
    'For spoken read-back completion, use the final report demonstrably after '
    'the relevant tool output. Do not use an early ASR interval spanning a '
    'lookup or write as timing proof, and never guess quote timing within it.'
)
def enabled(case):
    return (case.case_id.split('__')[0]==SCENE and
            case.task_contract_revision.get('medication_source_contract',{}).get('source_text_revision')==REVISION)

def note_from_arguments(args):
    return {'medicine':'Medicine A','reported_dose':args['documented_dose'],
            'source':args['source'],'source_participant_user_id':args['source_participant_user_id'],
            'source_document':args['source_document'],'confirmation_status':args['confirmation_status']}

def comparison_mapping(case,value):
    """Mask prose wording only inside a typed note/call; keep its semantic gate."""
    if not enabled(case) or not isinstance(value,dict):return value
    if 'existing_medication_list' in value and case.task_contract_revision['medication_source_contract'].get('legacy_flat_source_compatibility'):
        wrapped={'care_plans':{'CARE-MU-002':{'business_state':deepcopy(value)}}}
        value=project_legacy_source(case,wrapped)['care_plans']['CARE-MU-002']['business_state']
    identity={'source_document','source_participant_user_id','confirmation_status'}
    if (identity <= set(value) and ('documented_dose' in value or 'reported_dose' in value)
            and 'existing_medication_list' not in value and 'source' in value):
        value=dict(value)
        text=value['source']
        value['source']=('[provenance text: semantic completion required]'
                         if isinstance(text,str) and text.strip() else text)
    return value

def project_legacy_source(case,state):
    """Explicit offline compatibility for the old flat external-source mirror.

    Enabled only on a scoring copy of an actual v1-layout trace. It does not
    claim that a new tool result or schema was delivered to the SUT.
    """
    if not enabled(case) or not case.task_contract_revision['medication_source_contract'].get('legacy_flat_source_compatibility'):
        return state
    for rid,record in state.get('care_plans',{}).items():
        b=record.get('business_state',{})
        initial=case.environment.initial_state.get('care_plans',{}).get(rid,{}).get('business_state',{})
        if b.get('source')==initial.get('source'):continue
        matches=[n for n in b.get('external_medication_notes',[]) if isinstance(n,dict)
                 and n.get('source')==b.get('source')
                 and n.get('reported_dose')==b.get('documented_dose')
                 and all(n.get(k)==b.get(k) for k in ('source_document','source_participant_user_id','confirmation_status'))]
        if matches and 'source' in initial:b['source']=deepcopy(initial['source'])
    return state
