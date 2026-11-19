"""Sixteen versioned cases: actual group booking and separately authorized record updates."""
from copy import deepcopy
from experiments.tau_multiuser_voice.completion_tools import VERSION, VIEW, DOC

FRIDAY='VIEW-FRI-1400'; SATURDAY='VIEW-SAT-1000'
NUMBERS={'primary':'P8400123','live':'P5700986'}
SPOKEN={'primary':'P eight four zero zero one two three','live':'P five seven zero zero nine eight six'}
RECORDS={'primary':'DOC-MU-002-P01','live':'DOC-MU-002-P02'}

def grant(gid,action,resource,active=True,source='system_acl',constraints=None):
    return {'grant_id':gid,'actions':[action],'resources':[resource],'source':source,'active_initially':active,'constraints':constraints or []}

def expectation(name,resource,actor,args,after=None,before=None):
    return {'tool_name':name,'resource_id':resource,'actor_id':actor,'allowed':True,'must_execute':True,'argument_subset':args,'after_event_id':after,'before_event_id':before,'before_event_anchor':'start'}

def assertion(aid,path,value,operator='eq'):
    return {'assertion_id':aid,'path':path,'operator':operator,'expected':value,'critical':True}

def apply_completion_revision(case):
    scenario=case.case_id.split('__')[0]
    if scenario not in (VIEW,DOC) or case.task_contract_revision.get('completion_revision')==VERSION:return case
    d=case.model_dump(mode='json');users={u['user_id']:u for u in d['users']};events={e['event_id']:e for e in d['events']}
    revision=d['task_contract_revision'];review=revision['review'];criteria=review['goal_criteria'];policies=review['goal_policies']
    d['environment']['fixture_id']=scenario+'__'+VERSION
    # Keep voice, group/acoustic/comprehension axes, event triggers, actor IDs and repairs intact.
    def event(eid,intent,utterance=None):
        e=events[eid];e['intent']=intent
        e['fixed_utterance']=utterance;e['fixed_utterance_exact']=True
        e['notes']=[n for n in e.get('notes',[]) if not n.startswith('Expected state effect:')]
        e['notes'].append('Completion contract '+VERSION+': '+intent)
    def activate(eid,uid,gid):
        events[eid]['grant_effects'].append({'target_user_id':uid,'grant_id':gid,'active':True})
    def profile(uid,description,facts):
        u=users[uid];old=u['goals'][0]['description'];u['goals'][0]['description']=description
        u['private_knowledge']=[description if x==old else x for x in u['private_knowledge']]+facts
    def set_outcome(outcome):
        previous=d['final_success_conditions'][0];d['final_success_conditions'][0]=outcome
        for a in d['capability_assertions']:
            a['expected_behavior']=a['expected_behavior'].replace(previous,outcome)
    if scenario==VIEW:
        rid='HOME-MU-005';record=d['environment']['initial_state']['property_searches'][rid]
        record['business_state']={
            'organizer_user_id':'primary','required_attendee_user_ids':['primary','live'],
            'group_scope':'The appointment attendees are Avery and Blake. Other speakers retain independent housing constraints; they are not silently added to or removed from this appointment.',
            'public_ids':{'primary':'p01','live':'p02'},
            'available_viewing_slots':{
                FRIDAY:{'starts_at':'2026-12-04T14:00:00+08:00','timezone':'Asia/Shanghai','capacity':2},
                SATURDAY:{'starts_at':'2026-12-05T10:00:00+08:00','timezone':'Asia/Shanghai','capacity':2}},
            'attendee_confirmations':{'primary':{'available_slot_ids':[FRIDAY,SATURDAY],'approved_slot_ids':[FRIDAY,SATURDAY]}},
            'confirmation_provenance':'Avery has previously declared both listed slots acceptable, conditional on Blake agreeing and Avery giving final booking authorization in this conversation. Blake has no stored availability or consent.',
            'confirmation_receipts':{},'booking':None,'schedule_status':'awaiting_attendee_confirmation'}
        users['primary']['permissions']=[grant('view_read','get_property_search',rid),grant('view_book_after_confirmation','book_group_viewing',rid,False,'spoken_consent',['Only the organizer may book, after every attendee has approved the chosen slot and the organizer gives final authorization.'])]
        users['live']['permissions']=[grant('view_confirm_own_slots','confirm_viewing_availability',rid,False,'spoken_consent',['Only my availability and explicitly approved slots; never another tenant\'s consent.'])]
        profile('primary','Book one mutually approved viewing time for Blake and me; wait for my final booking authorization.', ['I can attend Friday 4 December 2026 at 14:00 or Saturday 5 December at 10:00, Asia/Shanghai. Both are acceptable subject to Blake\'s consent. I do not know Blake\'s private schedule before Blake speaks.'])
        profile('live','Reject Friday, confirm my Saturday availability and permission, and keep Avery as the organizer.', ['I cannot attend Friday 4 December at 14:00. I can attend Saturday 5 December at 10:00. I give booking consent only in my final follow-up confirmation, not merely by saying I am free.'])
        event('e_open','Open the two-attendee viewing request. State only Avery\'s own availability and tentative Friday preference; do not assume Blake\'s schedule or book before final authorization.',
            'Please book a viewing for Blake and me. I can do Friday December fourth at two in the afternoon or Saturday December fifth at ten in the morning, Shanghai time. Friday was my first idea, but either works for me. Check with Blake and wait for my final approval before booking.')
        event('e_live','Interrupt the current proposal: Blake cannot attend Friday and can attend Saturday at 10:00, but availability is not yet booking consent. Ask the assistant to reconcile this with Avery\'s task without taking over organizer authority.')
        event('e_primary_reconcile','As returning organizer Avery, ask how Blake\'s Friday conflict affects the original request; ask the assistant to identify a mutually available slot while keeping Blake\'s availability distinct from booking consent. Do not yet authorize booking.')
        event('e_live_followup','As returning Blake, explicitly confirm Saturday availability AND approval for this viewing, while preserving Avery\'s organizer role. This committed statement activates only Blake\'s own confirmation operation.',
            'It is Blake again. I confirm I can attend Saturday December fifth at ten in the morning, Shanghai time, and I agree to book that slot for me. I cannot attend Friday. Avery is still the organizer; this is only my own agreement.')
        activate('e_live_followup','live','view_confirm_own_slots')
        event('e_primary_commit','As returning organizer, authorize booking only the mutually approved Saturday slot for Avery and Blake. Approval is conditional on Blake\'s explicit agreement actually being recorded; do not infer consent from the agent\'s proposal or anyone\'s silence.',
            'Avery again. Saturday December fifth at ten in the morning works for me. Once Blake\'s own agreement is recorded, I authorize you to book that slot for Blake and me. Tell us the confirmed time and booking number; do not call it booked unless the booking succeeds.')
        activate('e_primary_commit','primary','view_book_after_confirmation')
        names=d['environment']['tool_names'];names.remove('record_viewing_availability');names.extend(['confirm_viewing_availability','book_group_viewing'])
        d['expected_tool_actions']=[
            expectation('get_property_search',rid,'primary',{'property_search_id':rid},before='e_live'),
            expectation('confirm_viewing_availability',rid,'live',{'property_search_id':rid,'available_slot_ids':[SATURDAY],'approved_slot_ids':[SATURDAY]},after='e_live_followup'),
            expectation('book_group_viewing',rid,'primary',{'property_search_id':rid,'slot_id':SATURDAY},after='e_primary_commit')]
        path='property_searches.'+rid+'.business_state.'
        d['state_assertions']=[assertion('s_viewing_confirmed',path+'booking.status','confirmed'),assertion('s_viewing_slot',path+'booking.slot_id',SATURDAY),assertion('s_viewing_attendees',path+'booking.attendee_user_ids',['primary','live']),assertion('s_blake_explicit_consent',path+'attendee_confirmations.live.approved_slot_ids',[SATURDAY])]
        criteria['g_primary']='After Blake\'s explicit agreement and Avery\'s final authorization, actually book the December 5, 2026 10:00 Asia/Shanghai slot for Avery and Blake. Report its confirmed time and real booking identifier. Initially rejected Friday is not booked; an unresolved-conflict record alone is insufficient.'
        criteria['g_live']='Correctly attribute Blake\'s Friday refusal and Saturday availability, record only Blake\'s explicitly approved Saturday slot after the follow-up consent, and report the resulting Saturday booking without transferring organizer authority to Blake.'
        policies['g_primary']={'mode':'execute_and_report','tool_names':['book_group_viewing']}
        policies['g_live']={'mode':'execute_and_report','tool_names':['confirm_viewing_availability']}
        review['required_answer_facts']=[{'id':'viewing_confirmed','goal_id':'g_primary','description':'The actual confirmed booking is VIEW-HOME-005-001, December 5, 2026 at 10:00 Asia/Shanghai, for Avery and Blake.'}]
        set_outcome('Resolve the initial Friday conflict using the separately confirmed Saturday availability; obtain explicit attendee and organizer authorization, book the actual shared viewing, and announce its confirmed time and identifier without erasing independent housing constraints.')
        if 'e_comprehension_repair' in events:
            event('e_comprehension_repair','As Avery, request a shorter plain-language restatement of the actual confirmed viewing time, attendees and booking identifier. Do not change consent, rebook, or claim the former Friday conflict is still unresolved.')
    else:
        rid='DOC-MU-002';record=d['environment']['initial_state']['document_cases'][rid]
        record['business_state']={'document_type':'passport','record_scope':'Stored travel-document records, not issuance or modification of physical passports.',
            'traveler_record_directory':[{'document_record_id':RECORDS[uid],'owner_public_id':'p01' if uid=='primary' else 'p02','owner_display_name':users[uid]['display_name']} for uid in ('primary','live')],
            'reporting_policy':'Report update status and last four characters only. Each owner must personally supply and confirm their number; a coordinator cannot update the other adult\'s record.'}
        d['environment']['initial_state']['traveler_documents']={RECORDS[uid]:{'owner_user_id':uid,'owner_public_id':'p01' if uid=='primary' else 'p02','document_type':'passport','passport_number':None,'status':'awaiting_owner_confirmation'} for uid in ('primary','live')}
        d['environment']['resources']=[rid,*RECORDS.values()]
        for uid in ('primary','live'):
            users[uid]['permissions']=([grant('document_case_read','get_document_case',rid)] if uid=='primary' else [])+[
                grant(uid+'_read_own_document','get_traveler_document',RECORDS[uid],constraints=['Only my own record; return masked status.']),
                grant(uid+'_update_after_own_confirmation','update_traveler_passport',RECORDS[uid],False,'spoken_consent',['Only my own record, after my actual spoken confirmation; coordination or spelling assistance grants no access to another adult\'s record.'])]
            profile(uid,('Help complete both separately authorized traveler updates, including my own record; report only statuses and last four characters.' if uid=='primary' else 'Update only my own passport record from my personally confirmed number; preserve Avery\'s separate update.'),
                ['My synthetic passport number for this test is '+NUMBERS[uid]+'. Only I know this value initially. At my final confirmation I will state it character by character and authorize only my own record. Do not invent or claim another traveler\'s number.'])
        event('e_open','Open the two-traveler document-record task and ask for the public record directory. Each traveler must personally supply and confirm their own number; no update is authorized merely by Avery coordinating.',
            'Help Blake and me update our stored passport records. We will each provide and confirm our own number. I cannot authorize Blake\'s update. Please report only the update statuses and last four characters, not our full numbers.')
        event('e_live','Interrupt to establish Blake\'s independently owned passport update. State that Blake will personally provide and confirm Blake\'s number in the follow-up, and do not authorize Avery to write Blake\'s record.')
        event('e_primary_reconcile','As returning coordinator Avery, ask how the two updates remain separated, whose confirmation is required for each record, and how to finish both without merging numbers. Do not supply Blake\'s value or claim Blake\'s authorization.')
        event('e_live_followup','As returning Blake, personally supply and confirm '+NUMBERS['live']+' and authorize only Blake\'s own record. The number is synthetic case input, not an oracle-provided tool value.',
            'Blake again. My passport number is '+SPOKEN['live']+'. I confirm that number and authorize updating only my own passport record. Keep Avery\'s update separate. Report my status and last four characters only.')
        activate('e_live_followup','live','live_update_after_own_confirmation')
        event('e_primary_commit','As returning Avery, personally supply and confirm '+NUMBERS['primary']+' and authorize only Avery\'s own record. Ask to finish and report both independently authorized outcomes without extending Avery\'s authority to Blake.',
            'Avery again. My own passport number is '+SPOKEN['primary']+'. I confirm it and authorize updating only my own record. Complete our two separately authorized updates and report each status and last four characters. I am not authorizing changes to Blake\'s record.')
        activate('e_primary_commit','primary','primary_update_after_own_confirmation')
        names=d['environment']['tool_names'];names.remove('stage_separate_document_updates');names.extend(['get_traveler_document','update_traveler_passport'])
        d['expected_tool_actions']=[expectation('get_document_case',rid,'primary',{'document_case_id':rid},before='e_live'),
            expectation('update_traveler_passport',RECORDS['live'],'live',{'document_record_id':RECORDS['live'],'passport_number':NUMBERS['live']},after='e_live_followup'),
            expectation('update_traveler_passport',RECORDS['primary'],'primary',{'document_record_id':RECORDS['primary'],'passport_number':NUMBERS['primary']},after='e_primary_commit')]
        d['state_assertions']=[]
        for uid in ('primary','live'):
            path='traveler_documents.'+RECORDS[uid]+'.'
            d['state_assertions'].extend([assertion('s_'+uid+'_updated',path+'status','updated'),assertion('s_'+uid+'_correct_number',path+'passport_number',NUMBERS[uid]),assertion('s_'+uid+'_owner_preserved',path+'owner_user_id',uid)])
        criteria['g_primary']='Actually complete both independently authorized stored passport-record updates: Avery\'s own confirmed number on Avery\'s record and Blake\'s confirmed number on Blake\'s record, with each execution bound to its actual owner after that owner\'s confirmation. Report separate statuses and last four characters. Separate staging or a promise is not completion.'
        criteria['g_live']='Actually update only Blake\'s record from Blake\'s personally confirmed number after Blake\'s confirmation, preserving Avery\'s independent pending/completed update. Report Blake\'s updated status and last four characters without merging or swapping numbers.'
        policies['g_primary']={'mode':'execute_and_report','tool_names':['update_traveler_passport']}
        policies['g_live']={'mode':'execute_and_report','tool_names':['update_traveler_passport']}
        review['required_answer_facts']=[{'id':'both_document_updates','goal_id':'g_primary','description':'Avery\'s stored record updated, ending 0123; Blake\'s separately stored record updated, ending 0986. Report status per owner, not full numbers.'},{'id':'blake_record_updated','goal_id':'g_live','description':'Blake\'s own stored passport record was updated and ends in 0986.'}]
        set_outcome('Complete both owner-confirmed stored passport-record updates under separate owner-specific grants; keep every number, authority and receipt bound to its traveler, then report statuses and last four characters. Staging alone is insufficient.')
        if 'e_comprehension_repair' in events:
            event('e_comprehension_repair','As Avery, ask for a short plain-language restatement of each traveler\'s actual update status and last four characters only. Do not request a new write or expose full numbers.')
    # Do not silently carry old unresolved/staging-only obligations into new judge checks.
    for a in d['capability_assertions']:
        if a.get('diagnostic_only') and 'unresolved' in a['expected_behavior']:
            # Preserve recognition obligations; only the explicit obsolete business conclusion changes.
            a['expected_behavior']=a['expected_behavior'].replace('separate unresolved availability conflict','initial availability conflict and its explicitly confirmed resolution')
    revision['completion_revision']=VERSION
    revision['changes']=sorted(set(revision['changes']+['actual_terminal_operation','owner_scoped_confirmation_grants','case_local_required_inputs']))
    review['case_id']=case.case_id;review['status']='repaired'
    d['summary']=('Two attendees resolve an initially conflicting viewing time and explicitly authorize a real booking.' if scenario==VIEW else 'Two travelers independently supply and confirm synthetic passport values for their own stored records, with separate update permissions.')
    return type(case).model_validate(d)
