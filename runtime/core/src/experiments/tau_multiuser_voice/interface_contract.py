"""Finite, documented parameter aliases. Never repairs resource digits or actor authority."""
from copy import deepcopy
import json
from pathlib import Path
ALIASES=json.loads(Path(__file__).with_name('interface_aliases_20261113.json').read_text())

def _normalize(value, aliases):
    if isinstance(value,str): return aliases.get(value,value)
    if isinstance(value,list): return [_normalize(x,aliases) for x in value]
    if isinstance(value,dict): return {k:_normalize(v,aliases) for k,v in value.items()}
    return value

def normalize_fields(scenario, arguments):
    out=deepcopy(arguments)
    for field,aliases in ALIASES.get(scenario,{}).items():
        if field in out: out[field]=_normalize(out[field],aliases)
    if scenario=='healthcare_care_02_medication_list_source':
        import re
        if isinstance(out.get('documented_dose'),str):
            m=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*(mg|g|mcg|mL)',out['documented_dose'])
            if m:out['documented_dose']=m[1]+' '+m[2]
        if out.get('confirmation_status')=='confirmation_required':out['confirmation_status']='patient_confirmation_required'
    return out

def expose_schema(schema,scenario):
    expose_nested_shapes(schema)
    if scenario=="healthcare_care_02_medication_list_source":
        schema["properties"]["documented_dose"].update(pattern=r"[0-9]+(?:\.[0-9]+)? (?:mg|g|mcg|mL)",description="Dose amount and unit only, such as a number followed by mg. Put attribution in source, not in the dose string.")
        schema["properties"]["confirmation_status"]["description"] += " confirmation_required is a finite alias for patient_confirmation_required in this API."
    for field,aliases in ALIASES.get(scenario,{}).items():
        prop=schema['properties'].get(field)
        if prop is None: continue
        note=' Accepted finite equivalents (alias -> canonical): '+json.dumps(aliases,ensure_ascii=False)+'. Other semantic values are not equated.'
        prop['description']=prop.get('description','')+note
        # Keep the domain open: these aliases do not enumerate every legal business value.
        # Publishing only the expected answer as an enum would leak the test solution.
    if 'until_condition' in schema['properties']:
        schema['properties']['until_condition']['description'] += ' Named completion conditions include active_emergency_video_call_finishes, active_transaction_finishes, and requester_confirms_ready; select the actual requested condition.'
    if scenario=='telecom_03_router_restart_pushback':schema['properties']['reason']['description'] += ' A faithful free-text explanation; no hidden reason code is required.'
    prop=schema['properties'].get('assignments_by_child')
    if prop is not None:
        prop['description']='Map each child name to the accepting driver: use the registered public participant ID (p01, p02, ...) or unique registered display name. No assignment or actor authority is inferred from a name.'
        prop['additionalProperties']={'type':'string'}


def expose_nested_shapes(schema):
    shapes={
      'votes_by_user':{'type':'string','enum':['support','oppose','abstain']},
      'answers_by_user':{'type':'string'},
      'gigabytes_by_user':{'type':'number','minimum':0},
      'items_by_owner':{'type':'array','items':{'type':'string'}},
      'excluded_items_by_owner':{'type':'array','items':{'type':'string'}},
      'accepted_roles_by_user':{'type':'array','items':{'type':'string'}},
      'rejected_roles_by_user':{'type':'array','items':{'type':'string'}},
      'reports_by_user':{'type':'string'},
      'timers_by_owner':{'type':'object','properties':{'label':{'type':'string'},'minutes':{'type':'number','exclusiveMinimum':0}},'required':['label','minutes'],'additionalProperties':False},
      'requests_by_user':{'type':'object','properties':{'amount':{'type':'number','minimum':0},'target_currency':{'type':'string'}},'required':['amount','target_currency'],'additionalProperties':False},
      'commutes_by_user':{'type':'object','properties':{'mode':{'type':'string'},'destination':{'type':'string'}},'required':['mode','destination'],'additionalProperties':False}}
    for field,shape in shapes.items():
        if field in schema.get('properties',{}):
            schema['properties'][field].update(type='object',additionalProperties=deepcopy(shape))
            schema['properties'][field]['description'] += ' Keys are registered public participant IDs. A single-owner request may supply only its own entry; independently supported entries are accumulated, so no compulsory batch call is required.'

def validate_shape(value,schema,label):
    import re
    kind=schema.get('type')
    types={'object':dict,'array':list,'string':str,'number':(int,float),'integer':int,'boolean':bool}
    if kind in types and (not isinstance(value,types[kind]) or (kind in {'number','integer'} and isinstance(value,bool))):raise ValueError('INVALID_ARGUMENT: wrong type for '+label)
    if 'enum' in schema and value not in schema['enum']:raise ValueError('INVALID_ARGUMENT: unsupported value for '+label)
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        if value<schema.get('minimum',float('-inf')) or value<=schema.get('exclusiveMinimum',float('-inf')):raise ValueError('INVALID_ARGUMENT: out of range '+label)
    if isinstance(value,dict):
        if set(schema.get('required',[]))-set(value):raise ValueError('INVALID_ARGUMENT: missing nested fields '+label)
        for k,v in value.items():
            field=schema.get('properties',{}).get(k,schema.get('additionalProperties',{}))
            if field is False:raise ValueError('INVALID_ARGUMENT: unknown nested field '+label+'.'+k)
            if isinstance(field,dict):validate_shape(v,field,label+'.'+k)
    if isinstance(value,list) and isinstance(schema.get('items'),dict):
        for v in value:validate_shape(v,schema['items'],label+'[]')
    if isinstance(value,str) and schema.get('pattern') and not re.fullmatch(schema['pattern'],value):raise ValueError('INVALID_ARGUMENT: invalid format for '+label)
