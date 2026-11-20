"""State transitions calculated from actual arguments, never reference answers."""
from copy import deepcopy

MAP_FIELDS={'gigabytes_by_user','requests_by_user','commutes_by_user','timers_by_owner',
 'items_by_owner','excluded_items_by_owner','votes_by_user','answers_by_user',
 'accepted_roles_by_user','rejected_roles_by_user','reports_by_user','assignments_by_child'}
def merge_arguments(state,args):
 aliases={"gigabytes_by_user":"data_pass_gigabytes_by_user","votes_by_user":"votes","answers_by_user":"answers","items_by_owner":"lists","reports_by_user":"reports"}
 for raw,canonical in aliases.items():
  if raw not in state and isinstance(state.get(canonical),dict):state[raw]=deepcopy(state[canonical])
 for k,v in args.items():
  if k in MAP_FIELDS and isinstance(v,dict):state.setdefault(k,{}).update(deepcopy(v))
  else:state[k]=deepcopy(v)

def derive(operation,args,state,actor):
 # OPERATION_ROWS contains explicit API completion statuses. Legacy resolution
 # payloads also contain target answers; none are copied into actual effects.
 from .business_tooling_v07 import OPERATION_ROWS
 out=deepcopy(OPERATION_ROWS[operation.scenario_id]['derived'])
 b=operation.scenario_id
 aliases={
  'telecom_01_family_data_lines':('data_pass_gigabytes_by_user','gigabytes_by_user'),
  'shared_home_03_grocery_threads':('lists','items_by_owner'),
  'meeting_01_overlap_votes':('votes','votes_by_user'),
  'education_01_fraction_answers':('answers','answers_by_user'),
  'emergency_incident_03_roll_call_overlap':('reports','reports_by_user')}
 if b in aliases:
  target,source=aliases[b];out[target]=deepcopy(state.get(source,{}))
 if b=='finance_billing_02_two_traveler_exchange':out['conversions']={u:[v['amount'],v['target_currency']] for u,v in state['requests_by_user'].items()}
 if b=='housing_location_02_roommate_commutes':out['commutes']={u:[v['mode'],v['destination']] for u,v in state['commutes_by_user'].items()}
 if b=='shared_home_01_parallel_timers':out['timers']={u:[v['label'],v['minutes']] for u,v in state['timers_by_owner'].items()}
 if b=='shared_home_03_grocery_threads':out['live_excluded_items']=deepcopy(state.get('excluded_items_by_owner',{}).get('live',[]))
 if b=='meeting_01_overlap_votes':
  voters=state.get('eligible_voter_ids');votes=out['votes']
  if voters is None:out['consensus']=bool(votes) and len(set(votes.values()))==1
  else:
   complete=all(u in votes for u in voters)
   out['consensus']=complete and all(votes[u]=='support' for u in voters)
   out['consensus_status']='reached' if out['consensus'] else 'not_reached' if complete or any(votes.get(u) in {'oppose','abstain'} for u in voters) else 'pending' 
 if b=='meeting_04_quiet_dissenter':
  votes=deepcopy(state.get('votes',{}));votes[args['participant_user_id']]=args['stance']
  out.update(votes=votes,consensus=bool(votes) and len(set(votes.values()))==1,floor_received=True)
  if 'live' in votes:out['live_position']=votes['live']
  if state.get('eligible_voter_ids'):
   voters=state['eligible_voter_ids'];complete=all(u in votes for u in voters)
   out['consensus']=complete and all(votes[u]=='support' for u in voters)
   out['consensus_status']='reached' if out['consensus'] else 'not_reached' if complete or any(votes.get(u) in {'oppose','abstain'} for u in voters) else 'pending' 
 if b=='healthcare_care_02_medication_list_source':
  from .medication_source_contract import note_from_arguments
  note=note_from_arguments(args)
  notes=deepcopy(state.get('external_medication_notes',[]))
  if note not in notes:notes.append(note)
  out.update(external_medication_notes=notes,verification_status='pending')
 if b=='education_01_fraction_answers':out['merge_speakers']=False
 if b=='education_04_project_roles':
  out.update(live_accepts=deepcopy(state.get('accepted_roles_by_user',{}).get('live',[])),live_rejects=deepcopy(state.get('rejected_roles_by_user',{}).get('live',[])))
 if b=='family_coordination_01_pickup_assignments':out['assignments']={**deepcopy(state['assignments_by_child']),**{c:'unresolved' for c in args['unresolved_children']}}
 if b=='family_coordination_03_chore_quiet_child' and args['family_member_user_id']=='live':out['live_existing_chore']=args['existing_chore']
 if b=='retail_02_shared_address_correction':out.update(corrected_house_number=args['house_number'],status=args['confirmation_status'])
 if b=='finance_billing_04_invoice_amount_revision':out['obsolete_amount_removed']=args['approved_amount']!=state.get('obsolete_approved_amount')
 if b in {'retail_04_return_side_talk','hospitality_03_table_side_talk'}:out['dessert_added']=state.get('dessert_added',False)
 if b=='telecom_03_router_restart_pushback':out['router_action']='paused'
 if b=='events_entertainment_01_ticket_ownership':out['assign_to_late_attendee']=not (args['attendee_user_id']=='live' and args['retained'])
 if b=='retail_05_cart_quantity_revision':out['item']=args['item_name']
 if b=='travel_identity_01_passport_digit_correction':out['corrected_candidate']=args['passport_number'];out['status']=args['verification_status']
 if b=='travel_identity_02_multi_traveler_documents':out['separate_owner_scopes']=deepcopy(args['document_owner_user_ids'])
 if b=='travel_identity_03_document_interpreter':out['source_user_id']=args['document_owner_user_id']
 if b=='telecom_02_lost_phone_number_fix':out={'line_'+args['lost_line_suffix']+'_status':'suspended','line_'+args['preserve_active_line_suffix']+'_status':'active'}
 if b=='telecom_04_roaming_interpreter':out['roaming_enabled']=args['enabled']
 if b=='vehicle_01_driver_destination':out.update(active_destination=args['destination'],route_authority_user_id=actor)
 if b=='meeting_05_result_due_date_update':out['task']=args['task_name']
 if b=='hospitality_03_table_side_talk' and args['guest_user_id']=='primary':out['primary_order']=args['entree']
 if b=='hospitality_04_bill_split' and args['payer_user_id']=='live':out['live_pays_items']=deepcopy(args['items'])
 if b=='public_services_01_transit_accessibility':out['preference_owner_user_id']=args['resident_user_id']
 if b=='it_support_01_service_owner_incident':out['until']=args['until_condition']
 if b=='it_support_03_rollback_correction':out.update(service=args['service_name'],version=args['target_version'],current_version=args['target_version'])
 if b=='airline_01_split_rebooking':out['rebooking_status']='partially_rebooked' if args['keep_original_user_ids'] else 'rebooked'
 if b=='airline_03_payer_traveler_cancel':out['cancellation_status']='cancelled'
 if b=='events_entertainment_01_ticket_ownership' and args['attendee_user_id']=='live':out['live_ticket_status']='retained' if args['retained'] else 'released'
 if b=='housing_location_01_couple_filter_conflict':out['location_filter']=None if args['downtown']=='preferred_not_mandatory' else 'downtown'
 return out
