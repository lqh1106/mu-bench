"""Versioned business resource-ID parsing, strictly before unchanged ACL checks.

Candidates come only from actual business records. No actor, permissions,
expected calls, goal dispositions, or evaluator state participates in resolution.
Only the declared primary resource field is eligible; other strings are untouched.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

VERSION = 'mu-business-resource-id-format-v1'
SEPARATORS = re.compile(r'[ \t\r\n\-\u2010\u2011\u2012\u2013\u2014\u2212]+')
SUPPORTED_FIELDS = {
    'booking_id': ('BK', 'bookings'), 'order_id': ('ORD', 'orders'),
    'service_account_id': ('TEL', 'service_accounts'), 'billing_case_id': ('BILL', 'billing_cases'),
    'property_search_id': ('HOME', 'property_searches'), 'document_case_id': ('DOC', 'document_cases'),
    'household_task_id': ('HSH', 'household_tasks'), 'trip_id': ('TRIP', 'trips'),
    'meeting_id': ('MTG', 'meetings'), 'learning_session_id': ('EDU', 'learning_sessions'),
    'care_plan_id': ('CARE', 'care_plans'), 'reservation_id': ('RSV', 'reservations'),
    'service_case_id': ('CIVIC', 'service_cases'), 'incident_id': ('INC', 'incidents'),
    'event_plan_id': ('EVT', 'event_plans'), 'family_plan_id': ('FAM', 'family_plans'),
    'incident_record_id': ('EMG', 'incident_records'),
}


def digits_key(digits):
    # String form avoids any integer conversion limit; leading zeros are a
    # formatting convention only for this explicitly declared numeric scheme.
    return digits.lstrip('0') or '0'


def resolve_id(raw, candidates, prefix):
    candidates = sorted(set(candidates))
    audit = dict(protocol=VERSION, raw_value=raw, canonical_value=None,
                 rule=None, status=None, candidate_count=0)
    def reject(code, explanation):
        audit.update(status='rejected',error_code=code)
        return None, audit, {'code':code,'message':explanation}
    if not isinstance(raw,str):
        return reject('RESOURCE_ID_INVALID_TYPE','Resource identifier must be a string containing its numeric identifier.')
    if raw in candidates:
        audit.update(status='exact',canonical_value=raw,rule='exact_record_identifier',candidate_count=1)
        return raw,audit,None
    token=SEPARATORS.sub('',raw).upper()
    if token in {'',prefix,prefix+'MU','MU'}:
        return reject('RESOURCE_ID_MISSING_INFORMATION','The resource number is missing. Supply the numeric identifier; it cannot be inferred from the template.')
    match=re.fullmatch(r'(?:'+re.escape(prefix)+r'MU|'+re.escape(prefix)+r'|MU)?([0-9]+)',token)
    if not match:
        return reject('RESOURCE_ID_INVALID_FORMAT','The identifier does not match this resource type. Preserve the object number; only case, separators and the declared template may vary.')
    key=digits_key(match.group(1))
    matches=[]
    for candidate in candidates:
        m=re.fullmatch(re.escape(prefix)+r'-MU-([0-9]+)',candidate)
        if m and digits_key(m.group(1))==key:matches.append(candidate)
    audit['candidate_count']=len(matches)
    if not matches:
        return reject('RESOURCE_ID_NOT_FOUND','No resource matches the supplied number. Check the number; do not substitute a different object.')
    if len(matches)!=1:
        return reject('RESOURCE_ID_AMBIGUOUS','This alias matches more than one record. Supply the complete exact identifier.')
    audit.update(status='normalized',canonical_value=matches[0],rule='declared_template_separators_case_and_numeric_zero_padding')
    return matches[0],audit,None


class BusinessResourceIDs:
    def __init__(self,environment):
        self.environment=environment
        self.contract=getattr(environment,'contract',None)
        self.supported=bool(self.contract and
            SUPPORTED_FIELDS.get(self.contract.resource_key)==(self.contract.id_prefix,self.contract.collection_key))

    def describe(self):
        return dict(protocol=VERSION,supported=self.supported,
            scope='declared_primary_business_resource_field_only',
            leading_zero_policy='equivalent_only_in_declared_PREFIX-MU-numeric_scheme',
            matching='exact_canonical_or_unique_numeric_alias',
            candidates_from='actual_business_records_not_acl_or_scoring_oracles',
            authorization='unchanged_after_resolution',
            identity_and_non_resource_arguments='unchanged')

    def prompt_suffix(self):
        if not self.supported:return ''
        field=self.contract.resource_key;prefix=self.contract.id_prefix
        return ('\nBusiness resource identifier format ('+VERSION+'):\n'
            f'- The {field} field uses {prefix}-MU-<number>. The object number is required.\n'
            '- Case, spaces/hyphens and numeric zero padding are formatting only. The fixed template may be omitted when this typed field uniquely identifies the record.\n'
            '- Never omit, guess, or replace the object number. Ambiguous aliases require the full exact identifier.\n'
            '- Identifier resolution does not grant permission; speaker binding and authorization still apply.\n')

    def resolve(self,tool_name,arguments):
        out=deepcopy(arguments)
        definition=self.environment.definitions.get(tool_name)
        if not self.supported or not definition or definition.resource_key!=self.contract.resource_key:
            return out,None,None
        field=self.contract.resource_key
        collection=self.environment.state.get(self.contract.collection_key,{})
        if not isinstance(collection,dict):raise ValueError('business resource registry is not a mapping')
        candidates=[]
        for key,record in collection.items():
            if not isinstance(key,str) or not isinstance(record,dict) or record.get(field)!=key:
                raise ValueError('business resource registry key/record mismatch')
            candidates.append(key)
        canonical,audit,error=resolve_id(out.get(field),candidates,self.contract.id_prefix)
        audit.update(field=field,tool_name=tool_name)
        if error:
            error=dict(error,field=field)
        else:
            out[field]=canonical
        return out,audit,error

    @staticmethod
    def record(output_dir,call_id,raw_arguments,arguments,audit,logical_ms):
        if audit is None:return
        row=dict(call_id=call_id,logical_ms=logical_ms,raw_arguments=raw_arguments,
                 normalized_arguments=arguments,resolution=audit)
        with (Path(output_dir)/'resource_id_resolution.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
