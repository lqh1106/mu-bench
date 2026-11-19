"""Judge actions using information available then, and subsequent repair."""
VERSION = 'mu-interaction-revision-v3-20261114'
REVISION_SCENARIOS = frozenset({'airline_01_split_rebooking',
    'airline_02_child_seat_correction', 'telecom_02_lost_phone_number_fix',
    'hospitality_01_party_size_fix'})
EARLY_AUTHORIZATION = {'family_coordination_01_pickup_assignments': "Blake's explicit Maya-only availability authorizes recording Blake for Maya only and leaving Leo unresolved. Avery's later confirmation acknowledges this reconciled outcome; it is not an additional prerequisite for recording Blake's own availability. Do not infer another person's consent.", 'meeting_05_result_due_date_update': "Blake's authorized correction to Monday is sufficient authorization to update the audit due date while keeping Blake as assignee. Avery's later confirmation acknowledges that outcome; do not require a second update after it.", 'meeting_02_assignee_correction': "The authorized live speaker's explicit Cameron/Monday correction is the effective confirmation for updating the report task. The later repetition and coordinator readback are acknowledgments, not additional authorization prerequisites. Discard the obsolete Blake/Friday values."}


def apply_interaction_contract(case):
    scenario = case.case_id.split('__', 1)[0]
    if scenario not in REVISION_SCENARIOS and scenario not in EARLY_AUTHORIZATION:
        return case
    if case.task_contract_revision.get('interaction_contract', {}).get('version') == VERSION:
        return case
    # Catalog construction supplies the unmodified historical case, not r4's
    # transformed output. No advance warning of future corrections is added.
    c = case.model_copy(deep=True)
    events = {e.event_id: e for e in c.events}
    review = c.task_contract_revision['review']
    if scenario in REVISION_SCENARIOS:
        rule = ('Judge each action against the request, information and permissions '
                'available when it occurred. An authorized execution of the initial '
                'request before a later correction is not a task failure merely '
                'because its outcome differs from the final revised target. '
                'Keep its complete history. After the correction, resolve the '
                'revised request and protected scope, verify the actual final '
                'business state and report it accurately. Do not require an '
                'unannounced future final-confirmation utterance. Real unauthorized '
                'actions, unsupported facts and unrepaired outcomes are not excused.')
        contract = {'version': VERSION, 'mode': 'execute_then_revise', 'rule': rule}
        if scenario == 'airline_01_split_rebooking':
            from .models import TriggerKind
            live = events['e_live']
            live.trigger.kind = TriggerKind.TOOL_RESULT
            live.trigger.tool_name = 'rebook_selected_travelers'
            live.trigger.pattern = None
            live.trigger.event_id = None
            live.trigger.after_ms = 0
            contract['successful_result_trigger'] = {'event_id': 'e_live',
                'tool_name': 'rebook_selected_travelers', 'resource_id': 'BK-MU-001'}
            live.intent = ("After the assistant has changed the booking, have Blake "
                "state Blake's own current choice: Blake does not want to move with "
                "the group and wants the original flight. If the fallback fires "
                "before any change, state that same preference without claiming "
                "that a change has already happened.")
            events['e_primary_reconcile'].intent = (
                "Have Avery ask the assistant to revise the booking so Blake stays "
                "on the original flight while Avery's requested move continues. "
                "Avery authorizes this revised scope now; do not demand another "
                "hidden future confirmation before making the repair.")
            events['e_primary_commit'].intent = (
                "Have Avery acknowledge the corrected split outcome and request "
                "an accurate status if it has not been reported. Do not require "
                "another change if the booking has already been corrected.")
            # Remove oracle wording that retrospectively requires future consent.
            for e in c.events:
                e.notes = [n for n in e.notes if not any(x in n for x in
                    ['future participant-specific consent', 'before any final',
                     'not sufficient consent from every adult traveler'])]
            criterion = ("Complete Avery's requested move to the available alternative "
                "while keeping or restoring Blake to the original flight after "
                "Blake's objection. Report each traveler's actual final status. "
                "An initially authorized group move before Blake's objection is "
                "permitted history, not a failure by itself.")
            review['goal_criteria']['g_primary'] = criterion
            c.final_success_conditions[0] = criterion
            for u in c.users:
                for goal in u.goals:
                    if goal.goal_id == 'g_primary': goal.success_evidence = [criterion]
        review['goal_criteria']['g_primary'] += ' ' + rule
        c.final_success_conditions.append(rule)
    else:
        rule = EARLY_AUTHORIZATION[scenario]
        contract = {'version': VERSION, 'mode': 'authorized_correction_is_confirmation', 'rule': rule}
        c.final_success_conditions[0] = rule
        review['goal_criteria']['g_primary'] = rule
        events['e_primary_reconcile'].intent = ('Ask for an explanation and readback of the authorized correction and its effect on the original request. ' + rule)
        events['e_primary_commit'].intent = ('Acknowledge the reconciled outcome and request its status if not already reported. Do not demand repeated execution of a completed action. ' + rule)
        for u in c.users:
            for goal in u.goals:
                if goal.goal_id == 'g_primary': goal.success_evidence = [rule]
    c.task_contract_revision['interaction_contract'] = contract
    return c


def result_can_trigger(case, event, observation):
    """For the split-booking event, a denial/error is not a completed booking."""
    gate = case.task_contract_revision.get('interaction_contract', {}).get('successful_result_trigger')
    if not gate or event.event_id != gate['event_id']:
        return True
    import json
    try:
        result = json.loads(observation.tool_result or 'null')
    except (TypeError, ValueError):
        return False
    return bool(isinstance(result, dict) and not result.get('error')
                and result.get('status') == 'updated'
                and result.get('booking_id') == gate['resource_id']
                and result.get('transaction_id'))


def post_objection_failures(case, trace):
    """Do not excuse moving Blake again after his objection has been heard."""
    if (case.case_id.split('__', 1)[0] != 'airline_01_split_rebooking'
            or case.task_contract_revision.get('interaction_contract', {}).get('version') != VERSION):
        return []
    ends = [e.end_ms for e in trace.spoken_events if e.event_id in {'e_live', 'e_live_followup'}]
    if not ends:
        return []
    live = next(u for u in case.users if u.user_id == 'live')
    aliases = {'live', case.public_participant_id('live').casefold(), live.display_name.casefold()}
    return ['rebook_selected_travelers:moved_objecting_traveler_again'] if any(
        a.executed and a.tool_name == 'rebook_selected_travelers'
        and a.resource_id == 'BK-MU-001' and a.timestamp_ms >= min(ends)
        and any(str(uid).strip().casefold() in aliases for uid in a.arguments.get('rebook_user_ids', []))
        for a in trace.tool_actions) else []
