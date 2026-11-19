"""Public participant IDs at the SUT boundary; internal evidence is unchanged."""
from __future__ import annotations
from copy import deepcopy
import re

VERSION = 'mu-business-public-participant-api-v3'
LABEL = 'PID-Public'

def field_kind(tool_name: str, field: str) -> str | None:
    if field.endswith('_user_id'): return 'value'
    if field.endswith('_user_ids'): return 'array_items'
    if field.endswith(('_by_user', '_by_owner')): return 'object_keys'
    if field == 'assignments_by_child': return 'object_values'
    return None


def configure_schemas(environment, tool_names):
    """Descriptions feed both native function schemas and the tool supplement."""
    for name in tool_names:
        definition = environment.definitions[name]
        definition.parameters = deepcopy(definition.parameters)
        for field, schema in definition.parameters.get('properties', {}).items():
            if field=='answers_by_user':
                schema['description']=schema.get('description','')+' Preserve each learner’s value separately. Exact rational strings and ordinary spoken fraction equivalents retain the same value; a different numerator or denominator is a different answer.'
            if field in {'wheelchair_access','step_free_entry'}:
                schema['description']=schema.get('description','')+' Use mandatory (required is an accepted equivalent) for a necessary constraint; optional is not equivalent.'
            kind = field_kind(name, field)
            if kind is None: continue
            location = {'value':'The value', 'array_items':'Each array item',
                        'object_keys':'Each object key', 'object_values':'Each object value'}[kind]
            note = (f'{location} is a registered public participant ID spoken during registration '
                    '(for example p01 or p02). Use that ID to select the participant; '
                    'it does not identify or authorize the requester. Other data in this field retain their meaning.')
            schema['description'] = schema.get('description','') + ' ' + note
        definition.description += (' Participant identifiers in returned records use registered public participant IDs '
                                   '(for example p01); these identifiers confer no authority or consent.')


def metadata(case, visible_tools):
    if not visible_tools: return None
    return {'protocol': VERSION, 'label': LABEL,
            'model_visible_namespace': 'registered_public_participant_id',
            'business_id_alias_table_exposed': False,
            'internal_execution_and_scoring_namespace_unchanged': True,
            'requester_source_unchanged': True,
            'public_id_argument_locations': {
                t['name']: {f:k for f in t['parameters'].get('properties',{})
                            if (k:=field_kind(t['name'],f))}
                for t in visible_tools if any(field_kind(t['name'],f) for f in t['parameters'].get('properties',{}))}}


def prompt_section(case, visible_tools):
    if not visible_tools: return ''
    return '''
Business participant identifier protocol (PID-Public):
- Use the public participant IDs spoken during registration, such as p01 and p02, in business parameters whose schema identifies a participant. Those parameter descriptions specify whether the ID is a value, array item, object key or object value.
- Participant references in tool results use the same public IDs. Associate them with voices and names using the spoken introductions; there is no separate business identity to infer.
- An operation's participant parameter selects its subject, not its requester. The actual requester is separately supplied in caller_id and checked for permission. Attribution reports are recorded predictions, not the requester binding. A participant appearing in a record is not automatically a target or a consenting party.
- Keep resource identifiers, child names, free text, numbers, dates and other non-participant data unchanged according to their own schemas.
'''


def public_result(case, tool_name: str, result):
    """Project typed identity positions only, never arbitrary narrative text.

    Called only on the copy sent to the model, after execution/authorization.
    The original result, state and oracle trace retain internal IDs.
    """
    ids = {u.user_id:case.public_participant_id(u.user_id) for u in case.users}
    def scalar(v): return ids.get(v,v) if isinstance(v,str) else v
    scalar_fields = {'actor_id','requested_by','last_modified_by','user_id',
                     'owner_id','speaker_id','requester_id'}
    list_fields = {'requested_scope','participant_ids','eligible_voter_ids'}
    key_fields = {'public_ids','attendee_confirmations','timers','lists','votes','answers','reports'}
    def visit(value, field='', path=()):
        kind=field_kind(tool_name,field)
        if field=='assignments':kind='object_values'
        if isinstance(value,str):
            if kind=='value' or field in scalar_fields: return scalar(value)
            # Existing machine-readable identity denial; preserve the decision
            # and wording, translating only its two explicit identity slots.
            if field=='error' and value.startswith('speaker/actor mismatch: claimed '):
                match=re.fullmatch(r'speaker/actor mismatch: claimed (\S+), actual (\S+)',value)
                if match:return f'speaker/actor mismatch: claimed {scalar(match[1])}, actual {scalar(match[2])}'
            return value
        if isinstance(value,list):
            if kind=='array_items' or field in list_fields:return [scalar(x) if isinstance(x,str) else visit(x,'',path+('[]',)) for x in value]
            return [visit(x,'',path+('[]',)) for x in value]
        if isinstance(value,dict):
            keys_are_ids = kind=='object_keys' or field in key_fields
            # HOME-MU-002 stores per-person commute data directly under
            # business_state; other business keys keep their spelling.
            scoped_business_keys = field=='business_state'
            out={}
            aliases=case.task_contract_revision.get('public_result_field_aliases',{})
            for k,v in value.items():
                public_field=aliases.get(k,k) if field=='business_state' else k
                new_key=scalar(k) if keys_are_ids or (scoped_business_keys and k in ids and isinstance(v,dict)) else public_field
                if new_key in out:raise ValueError('ambiguous public participant keys in tool result')
                if kind=='object_values':out[new_key]=scalar(v) if isinstance(v,str) else visit(v,k,path+(k,))
                elif keys_are_ids:out[new_key]=visit(v,'',path+(k,))
                else:out[new_key]=visit(v,k,path+(k,))
            return out
        return deepcopy(value)
    return visit(result)
