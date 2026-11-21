"""T-R4/JT6: field-scoped format equivalence, never reference-answer repair."""
from copy import deepcopy
import os
import re

VERSION = 'mu-business-values-v1'
# Rules identify API fields, not expected answers. Unknown fields and IDs stay exact.
RULES = {
 'airline_02_child_seat_correction': {'seat_preference':'label'},
 'education_04_project_roles': {'accepted_roles_by_user.*.[]':'label','rejected_roles_by_user.*.[]':'label','live_accepts.[]':'label','live_rejects.[]':'label'},
 'emergency_incident_02_lost_child_description': {'jacket_color':'label'},
 'emergency_incident_03_roll_call_overlap': {'reports_by_user.*':'report','reports.*':'report'},
 'events_entertainment_05_schedule_time_fix': {'doors_time':'time','show_time':'time'},
 'family_coordination_01_pickup_assignments': {'unresolved_children.[]':'label'},
 'finance_billing_02_two_traveler_exchange': {'target_currency':'currency','source_currency':'currency','requests_by_user.*.target_currency':'currency','conversions.*.1':'currency'},
 'finance_billing_04_invoice_amount_revision': {'currency':'currency'},
 'healthcare_care_01_appointment_consent': {'appointment_slot':'label'},
 'hospitality_03_table_side_talk': {'entree':'label','primary_order':'label'},
 'hospitality_04_bill_split': {'items.[]':'label','live_pays_items.[]':'label'},
 'housing_location_02_roommate_commutes': {'mode':'label','destination':'label','commutes_by_user.*.mode':'label','commutes_by_user.*.destination':'label','commutes.*.0':'label','commutes.*.1':'label'},
 'meeting_02_assignee_correction': {'due_date':'weekday'},
 'meeting_04_quiet_dissenter': {'stance':'label','votes.*':'label','live_position':'label'},
 'meeting_05_result_due_date_update': {'task_name':'label','task':'label','due_date':'weekday'},
 'public_services_02_appointment_interpreter': {'slot':'slot','available_appointment_slots.[]':'slot'},
 'retail_02_shared_address_correction': {'street':'label'},
 'retail_04_return_side_talk': {'item_condition':'label','refund_destination':'refund'},
 'retail_05_cart_quantity_revision': {'item_name':'label','item':'label'},
 'shared_home_01_parallel_timers': {'timers_by_owner.*.label':'label','timers.*.0':'label','primary_timer.0':'label','live_timer.0':'label','existing_timer.0':'label'},
 'shared_home_02_media_backchannel': {'album':'label'},
 'shared_home_03_grocery_threads': {'items_by_owner.*.[]':'label','excluded_items_by_owner.*.[]':'label','lists.*.[]':'label','live_excluded_items.[]':'label'},
 'telecom_04_roaming_interpreter': {'region':'label'},
 'vehicle_01_driver_destination': {'destination':'label','active_destination':'label'},
}
WEEKDAYS = {x.casefold():x for x in ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')}
REPORTS = {'all accounted for':'all_accounted_for','one unaccounted for':'one_unaccounted_for','one person unaccounted for':'one_unaccounted_for'}
REFUNDS = {'original payment method':'original_payment_method'}

def enabled(case=None):
 mode=os.environ.get('MUBENCH_BUSINESS_VALUE_CONTRACT')
 if mode=='off':return False
 return mode==VERSION or os.environ.get('MUBENCH_TOOL_JUDGE_VERSION') in {'jt6','jt7'} or (case is not None and case.task_contract_revision.get('business_value_contract')==VERSION)

def _rule(scene,path):
 for pattern,kind in RULES.get(scene,{}).items():
  parts=pattern.split('.')
  if len(path)>=len(parts) and all(p=='*' or p==str(v) for p,v in zip(parts,path[-len(parts):])):return kind
 return None

def _time(value):
 # Explicit AM/PM may be converted; bare hours retain their supplied time basis.
 m=re.fullmatch(r'(\d{1,2}):(\d{2})\s*(am|pm)?',value,re.I)
 if not m or int(m[2])>=60:return value
 hour=int(m[1]);suffix=m[3]
 if suffix:
  if not 1<=hour<=12:return value
  hour=hour%12+(12 if suffix.casefold()=='pm' else 0)
 elif hour>=24:return value
 return f'{hour:02d}:{m[2]}'

def scalar(kind,value):
 text=' '.join(value.strip().split())
 if kind=='label':return text.casefold()
 if kind=='currency':return text.upper() if re.fullmatch('[A-Za-z]{3}',text) else value
 if kind=='weekday':return WEEKDAYS.get(text.casefold(),value)
 if kind=='time':return _time(text)
 if kind=='slot':
  m=re.fullmatch(r'([A-Za-z]+)\s+(\d{1,2}:\d{2})',text)
  if m and m[1].casefold() in WEEKDAYS:return WEEKDAYS[m[1].casefold()]+' '+_time(m[2])
  return value
 if kind in {'report','refund'}:
  table=REPORTS if kind=='report' else REFUNDS
  key=re.sub(r'[_\s-]+',' ',text).casefold()
  return table.get(key,value)
 return value

def canonical(scene,value,path=()):
 if isinstance(value,dict):
  child_names=scene=='family_coordination_01_pickup_assignments' and path and path[-1] in {'assignments_by_child','assignments'}
  out={}
  for k,v in value.items():
   key=scalar('label',k) if child_names and isinstance(k,str) else k
   if key in out:raise ValueError('ambiguous child names after case-insensitive normalization')
   out[key]=canonical(scene,v,path+(k,))
  return out
 if isinstance(value,list):
  # Numeric positions are preserved for derived [label, number] records;
  # [] patterns match array elements through the second lookup below.
  out=[]
  for i,v in enumerate(value):
   kind=_rule(scene,path+('[]',))
   out.append(scalar(kind,v) if isinstance(v,str) and kind else canonical(scene,v,path+(str(i),)))
  return out
 if isinstance(value,str):return scalar(_rule(scene,path),value)
 return deepcopy(value)

def normalize_arguments(scene,value):
 # Preserve the public spelling of child names in actual tool results; only
 # the comparison copy resolves case-insensitive child-name keys.
 if scene=='family_coordination_01_pickup_assignments':return deepcopy(value)
 return canonical(scene,value) if enabled() else deepcopy(value)

def annotate_schema(schema,scene):
 """Publish representation rules, not the expected value or speaker mapping."""
 top_fields={p.split('.')[0] for p in RULES.get(scene,{})}
 for field,prop in schema.get('properties',{}).items():
  if scene=='family_coordination_01_pickup_assignments' and field=='assignments_by_child':
   prop['description']=prop.get('description','')+' Child-name keys resolve case/whitespace variants only to one actual name in the public family record; no fuzzy name correction. Participant-ID values retain their identity and confer no authority.'
  if field not in top_fields:continue
  notes={
   'label':'Text labels ignore letter case and outer/repeated whitespace. Distinct objects, words, qualifiers, owners and constraints remain distinct; no fuzzy spelling repair.',
   'currency':'Use a three-letter currency code; letter case is ignored. Different currencies remain different.',
   'weekday':'Use an English weekday name, case-insensitively, or the actual explicit date. Dates are not inferred from weekday names.',
   'time':'Use local 24-hour HH:MM (00:00-23:59), using the daypart actually supplied in the event information or request. An explicit AM/PM spelling is converted equivalently. Leading hour zero is optional. Bare 06:00 and 18:00 remain distinct; no daypart, date or timezone is inferred.',
   'slot':'Use the actual returned weekday and HH:MM slot. Weekday case and hour zero padding are equivalent; other slots are not.',
   'report':'Known equivalent encodings include all_accounted_for / all accounted for, and one_unaccounted_for / one unaccounted for / one person unaccounted for. These distinguish zero missing from exactly one missing; other reports are not guessed.',
   'refund':'original_payment_method and original payment method identify the same refund channel. Other destinations remain distinct.',
  }
  kinds={kind for path,kind in RULES[scene].items() if path.split('.')[0]==field}
  note=' '.join(notes[k] for k in sorted(kinds))
  prop['description']=prop.get('description','')+' '+note
  if field=='seat_preference':prop['description']+=' State the seat-position preference only (for example aisle, window, or middle). Do not bury a separate adjacency constraint inside this value or claim adjacency was executed by this endpoint.'
  if field=='appointment_slot':prop['description']+=' State the actual requested appointment period; morning and afternoon are not equivalent.'
  if field=='stance':prop['description']+=' Vote labels are support, oppose and abstain, with different meanings.'
  if field in {'album','street','unresolved_children','item_name'}:prop['description']+=' Select the object/name actually supplied in the request or public record. Do not add or remove title words or silently repair identifying digits.'
  if field=='album':prop['description']+=' The album field contains the title only; full-album playback belongs in playback_scope, not in the title.'
  if field in {'accepted_roles_by_user','rejected_roles_by_user'}:prop['description']+=' Keep preparation of slides, speaking/presentation, and research as distinct roles; an accepted role is never inferred from a rejection or vice versa.'


def resolve_public_child_names(arguments, children):
 """Resolve case/space variants against the actual resource, not an oracle."""
 def resolve(value):
  if not isinstance(value,str):return value
  hits=[name for name in children if isinstance(name,str) and scalar('label',name)==scalar('label',value)]
  if len(hits)>1:raise ValueError('INVALID_ARGUMENT: child name is ambiguous in the public record')
  return hits[0] if hits else value
 out=deepcopy(arguments)
 if isinstance(out.get('unresolved_children'),list):out['unresolved_children']=[resolve(v) for v in out['unresolved_children']]
 if isinstance(out.get('assignments_by_child'),dict):
  mapped={}
  for k,v in out['assignments_by_child'].items():
   name=resolve(k)
   if name in mapped:raise ValueError('INVALID_ARGUMENT: duplicate child identity')
   mapped[name]=v
  out['assignments_by_child']=mapped
 return out
