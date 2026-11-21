"""Case-local completion APIs for viewing bookings and owner-scoped passport records.

These APIs consume runner-bound actors. Spoken-event ACL transitions remain in
AuthorizationOracle; no model-provided boolean can create a permission grant.
"""
from copy import deepcopy
from inspect import Parameter, Signature
import re
from typing import Any
from experiments.tau_multiuser_voice.virtual_tools import VirtualToolDefinition

VERSION = 'completion-20261113-r2'
VIEW = 'housing_location_05_group_viewing_schedule'
DOC = 'travel_identity_02_multi_traveler_documents'
TOOL_NAMES = {'confirm_viewing_availability','book_group_viewing','get_traveler_document','update_traveler_passport'}

def definition(name, description, fields, resource_key, effect, lookup):
    return VirtualToolDefinition(name=name, description=description, effect=effect,
        parameters={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False},
        returns={'type':'object','description':'Actual operation outcome, stable resource/transaction identifiers, and owner-scoped status. Passport results disclose last four characters only.'},
        resource_key=resource_key, requester_source='runner_bound_speaker_context', mutates_state=effect=='write',
        call_method='POST /v2/'+name, result_lookup=lookup,
        idempotency='Repeating the identical effective request preserves the existing result and creates no duplicate transaction.',
        errors=['RESOURCE_NOT_FOUND: unknown exact resource or slot.', 'AUTHORIZATION_DENIED: wrong bound owner or organizer.', 'CONFLICT: missing agreement or incompatible slot.', 'INVALID_ARGUMENT: malformed input.'])

def definitions(binding):
    if not (binding.fixture_id or '').endswith(VERSION):return {}
    string=lambda desc:{'type':'string','description':desc}
    array=lambda desc:{'type':'array','items':{'type':'string'},'description':desc}
    if binding.fixture_id.startswith(VIEW):
        return {d.name:d for d in [
            definition('confirm_viewing_availability',
                'Record only the actual speaking attendee\'s available slots and separately approved booking slots. Availability alone is NOT consent. Use slot identifiers returned by get_property_search; never supply another person as actor.',
                {'property_search_id':string('Property search identifier.'),'available_slot_ids':array('All slots this attendee explicitly says they can attend.'),'approved_slot_ids':array('Subset this attendee explicitly authorizes for booking; [] means no booking consent.')},
                'property_search_id','write','get_property_search'),
            definition('book_group_viewing',
                'Book one viewing for every required attendee of this property search. Only the bound organizer may request it after final spoken authorization; every attendee must have explicitly approved the selected slot. A helper\'s suggestion is not consent.',
                {'property_search_id':string('Property search identifier.'),'slot_id':string('One available and jointly approved slot identifier from get_property_search.')},
                'property_search_id','write','get_property_search')
        ]}
    if binding.fixture_id.startswith(DOC):
        return {d.name:d for d in [
            definition('get_traveler_document','Read only the bound traveler\'s own document-record status and last four passport characters. The case summary provides the public record-to-owner mapping.',
                {'document_record_id':string('Exact traveler record identifier from get_document_case; different records have different permissions.')},
                'document_record_id','read','get_traveler_document'),
            definition('update_traveler_passport','Update only the bound traveler\'s own stored passport record, after that traveler has supplied and explicitly confirmed their number. A coordinator cannot update another adult\'s record. Letters are case-insensitive; spaces and hyphens are formatting only; all letters and digits must otherwise match exactly. The result reveals only last four characters.',
                {'document_record_id':string('Exact owner-specific record ID from get_document_case.'),'passport_number':string('Owner-supplied synthetic passport number; 6–12 ASCII letters/digits after removing spaces and hyphens.')},
                'document_record_id','write','get_traveler_document')
        ]}
    return {}

def signature(definition):
    types={'string':str,'array':list[str],'boolean':bool,'integer':int}
    return Signature([Parameter(k,Parameter.POSITIONAL_OR_KEYWORD,annotation=types[v['type']]) for k,v in definition.parameters['properties'].items()],return_annotation=dict[str,Any])

def normalize(arguments, tool_name):
    out=deepcopy(arguments)
    # Formatting tolerance is lossless for identifier letters/digits, not fuzzy correction.
    key='document_record_id' if tool_name in ('get_traveler_document','update_traveler_passport') else 'property_search_id'
    if isinstance(out.get(key),str):
        # The actual resource lookup resolves only an unambiguous full identifier.
        out[key]=out[key].strip()
    if tool_name=='update_traveler_passport' and isinstance(out.get('passport_number'),str):
        out['passport_number']=re.sub(r'[\s-]','',out['passport_number']).upper()
    return out

def resolve_record_id(state, supplied):
    """Tolerate separators, never discard/change semantic letters or digits."""
    if not isinstance(supplied,str):raise ValueError('INVALID_ARGUMENT: resource ID must be a string')
    token=lambda x:re.sub(r'[\s_-]','',x).upper()
    matches=[k for k in state if token(k)==token(supplied)]
    if len(matches)!=1:raise ValueError('RESOURCE_NOT_FOUND: unknown or ambiguous exact record ID')
    return matches[0]

def _write(env, tool_name, actor, resource_id, arguments, result):
    # Effective repeat is handled by each operation before reaching this function.
    event={'transaction_id':result['transaction_id'],'tool_name':tool_name,'resource_id':resource_id,
           'actor_id':actor,'arguments':deepcopy(arguments),'result':deepcopy(result)}
    env.state.setdefault('audit_events',[]).append(event)
    return deepcopy(result)

def execute(env, tool_name, actor, arguments):
    args=normalize(arguments,tool_name)
    expected=set(env.definitions[tool_name].parameters['properties'])-{'caller_id'}
    if set(args)!=expected:raise ValueError('INVALID_ARGUMENT: expected fields '+str(sorted(expected)))
    if tool_name in ('confirm_viewing_availability','book_group_viewing'):
        rid=resolve_record_id(env.state['property_searches'],args['property_search_id'])
        record=env.state['property_searches'][rid];state=record['business_state']
        attendees=state['required_attendee_user_ids'];organizer=state['organizer_user_id'];slots=state['available_viewing_slots']
        if tool_name=='confirm_viewing_availability':
            if actor not in attendees:raise ValueError('AUTHORIZATION_DENIED: speaker is not a booked attendee')
            available=args['available_slot_ids'];approved=args['approved_slot_ids']
            if not isinstance(available,list) or not isinstance(approved,list) or any(not isinstance(x,str) for x in available+approved):raise ValueError('INVALID_ARGUMENT: slot arrays required')
            if len(set(available))!=len(available) or len(set(approved))!=len(approved):raise ValueError('INVALID_ARGUMENT: duplicate slot IDs')
            if set(available)-set(slots):raise ValueError('RESOURCE_NOT_FOUND: unknown available slot')
            if set(approved)-set(available):raise ValueError('CONFLICT: approval outside declared availability')
            if state.get('booking') and state['booking']['slot_id'] not in approved:raise ValueError('CONFLICT: existing booking requires explicit cancellation before withdrawing approval')
            new={'available_slot_ids':sorted(available),'approved_slot_ids':sorted(approved)}
            if state['attendee_confirmations'].get(actor)==new:
                return deepcopy(state['confirmation_receipts'][actor])
            state['attendee_confirmations'][actor]=new
            tx=f'TX-VIEW-CONFIRM-{actor}-{len(env.state["audit_events"])+1}'
            result={'property_search_id':rid,'transaction_id':tx,'attendee_public_id':state['public_ids'][actor],**new,'status':'attendee_confirmation_recorded'}
            state['confirmation_receipts'][actor]=deepcopy(result)
        else:
            if actor!=organizer:raise ValueError('AUTHORIZATION_DENIED: only the actual organizer may book')
            slot=args['slot_id']
            if slot not in slots:raise ValueError('RESOURCE_NOT_FOUND: unknown viewing slot')
            if slots[slot]['capacity']<len(attendees):raise ValueError('CONFLICT: insufficient appointment capacity')
            for uid in attendees:
                consent=state['attendee_confirmations'].get(uid,{})
                if slot not in consent.get('available_slot_ids',[]) or slot not in consent.get('approved_slot_ids',[]):
                    raise ValueError('CONFLICT: missing availability or explicit approval for '+state['public_ids'][uid])
            if state.get('booking'):
                if state['booking']['slot_id']!=slot:raise ValueError('CONFLICT: a different booking already exists')
                return deepcopy(state['booking_receipt'])
            booking={'booking_id':'VIEW-HOME-005-001','slot_id':slot,'starts_at':slots[slot]['starts_at'],'timezone':slots[slot]['timezone'],'attendee_user_ids':list(attendees),'status':'confirmed'}
            state['booking']=booking;state['schedule_status']='booked'
            result={'property_search_id':rid,'transaction_id':'TX-VIEW-BOOK-001',**booking,'attendee_public_ids':[state['public_ids'][u] for u in attendees]}
            # Tool returns public participant references, never internal speaker labels.
            result.pop('attendee_user_ids')
            state['booking_receipt']=deepcopy(result)
        record.update(status='updated',last_operation=tool_name,last_modified_by=actor,last_transaction_id=result['transaction_id'])
        return _write(env,tool_name,actor,rid,args,result)
    records=env.state['traveler_documents']
    rid=resolve_record_id(records,args['document_record_id']);record=records[rid]
    if record['owner_user_id']!=actor:raise ValueError('AUTHORIZATION_DENIED: a traveler may access only their own record')
    if tool_name=='get_traveler_document':
        return {'document_record_id':rid,'owner_public_id':record['owner_public_id'],'status':record['status'],'last_four':record['passport_number'][-4:] if record['passport_number'] else None,'transaction_id':None}
    number=args['passport_number']
    if not isinstance(number,str) or not re.fullmatch(r'[A-Z0-9]{6,12}',number):raise ValueError('INVALID_ARGUMENT: passport requires 6–12 ASCII letters/digits')
    if record['passport_number']==number and record['status']=='updated':return deepcopy(record['update_receipt'])
    result={'document_record_id':rid,'owner_public_id':record['owner_public_id'],'status':'updated','last_four':number[-4:],'transaction_id':'TX-DOC-'+record['owner_public_id'].upper()+'-'+str(len(env.state['audit_events'])+1)}
    record.update(passport_number=number,status='updated',update_receipt=deepcopy(result))
    return _write(env,tool_name,actor,rid,args,result)

def _completion_comparable_state(case,state):
    """Compare final business effects, preserving the full real audit log elsewhere.

    Only these new contracts are order-independent: legitimate retries/owner
    updates may have different audit order/transaction IDs. Identity, permission,
    argument and temporal checks still run on every original ToolActionTrace.
    Unknown state fields are retained, so unrelated writes cannot disappear.
    """
    if case.task_contract_revision.get('completion_revision')!=VERSION:return state
    out=deepcopy(state);out.pop('audit_events',None)
    for collection in ('property_searches','document_cases'):
        for record in out.get(collection,{}).values():
            for key in ('last_operation','last_modified_by','last_transaction_id'):record.pop(key,None)
            business=record.get('business_state',{})
            for key in ('confirmation_receipts','booking_receipt'):business.pop(key,None)
    for record in out.get('traveler_documents',{}).values():record.pop('update_receipt',None)
    return out


def comparable_state(case, state):
    from .evaluation_policy import business_state_projection
    result = business_state_projection(case, _completion_comparable_state(case, state))
    if case.task_contract_revision.get("design_revision") in {"tool-scene-r2","tool-scene-r3","tool-scene-r4","tool-scene-r5"} or __import__("os").environ.get("MUBENCH_TOOL_CASE_MECHANISM") in {"tr2","tr3","tr4","tr5"}:
        from .business_acceptance import project
        return project(case, result)
    from .business_values import enabled, canonical
    return canonical(case.case_id.split('__')[0],result) if enabled(case) else result
