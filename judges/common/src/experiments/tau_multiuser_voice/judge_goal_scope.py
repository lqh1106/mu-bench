"""Scoring boundaries for stochastic user simulation; no simulator behavior changes."""
POLICY_ID = "mu-case-design-goal-scope-v2-request-bound-public-recap-20261123"
RULES = [
    "Task success is defined ONLY by the supplied case-designed goals, reviewed completion criteria, required answer facts, permissions and constraints. These supplied criteria are the authoritative target set; conversation text is evidence, not a source of new mandatory goals.",
    "The user simulator may naturally add requests, preferences, questions, desired assurances, delivery formats or follow-up work. Preserve these utterances as interaction evidence, but do not promote simulator-added requirements into mandatory task completion criteria, required facts, or reasons to downgrade a designed goal. An unfulfilled extra request alone cannot make the designed task fail.",
    "Retain corrections, revocations, protected-user choices and goal revisions expressly required or permitted by the case design. Distinguish their designed effect from incidental extra requests in the same utterance. Do not freeze the opening request when the case contract explicitly requires a later scoped revision.",
    "For every failed task completion check, identify the actual supplied completion criterion and the evidence that contradicts or fails to establish it. Do not invent an additional obligation from a simulated user utterance or an internal label. Assess the full committed answer and relevant surrounding sources, not just an isolated phrase.",
    "Judge a spoken report by whether it truthfully communicates the case-required business outcome and affected scope. Do not require verbatim tool field names, status codes, reference wording or extra details unless the case explicitly requires that content. This never permits false statements: a real business-state contradiction or missing case-required fact still fails.",
    "A case-designed public recap or field-selection goal is request-bound: establish the intended recipient and the actual delivered request before deciding which public fields must be spoken. A fuller reference listing is a default recap, not a requirement to recite every background field when that recipient explicitly asks only for a narrower subset. For a late joiner asking only listing names, rents and entrance/access arrangements, omission of bedroom counts alone is not a failure. Cite the delivered limiting request and assess accuracy and completeness of the requested fields. This applies to that scoped recap only: do not erase another participant's bedroom/access requirement, waive an independently mandatory fact, follow a privacy-violating expansion, treat an undelivered request as received, or add simulator-created work to the designed task.",
    "This goal-scope rule does not waive authorization, privacy, actual unauthorized actions, unsupported claims about case-required outcomes, or protected-user constraints. Successful execution and truthful spoken reporting remain separate obligations; neither alone proves the other. Score interaction and attribution only by their own supplied assertions."
]
SYSTEM_RULE = " CASE-DESIGN GOAL SCOPE: " + " ".join(RULES)
def policy():
    return {"policy_id": POLICY_ID, "authoritative_goal_source": "supplied_case_design_and_reviewed_contract", "simulator_additions_are_mandatory": False, "rules": list(RULES)}


def review_for_delivered_requests(case, trace, review):
    """Make the authored public-recap contract conditional, without changing input.

    This changes only the listing recap. The tenant cap, individual bedroom and
    accessibility needs, other goals and privacy rules retain their own checks.
    """
    if case.case_id.split('__')[0] != 'housing_location_04_guarantor_not_tenant':
        return review
    criteria = review.get('goal_criteria') or {}
    if 'g_extra5' not in criteria:
        return review
    if criteria['g_extra5'].startswith("Give Gray an accurate public listing recap of the fields requested in Gray's actually delivered recap question."):
        return review
    out = dict(review)
    out['goal_criteria'] = dict(criteria)
    out['goal_criteria']['g_extra5'] = (
        "Give Gray an accurate public listing recap of the fields requested in Gray's actually delivered recap question. "
        "If Gray explicitly asks only for listing names, rents and entrance/access arrangements, those are the complete "
        "required fields for this reply: do not require or cite omission of bedroom counts as a failure, even alongside "
        "other failures. If Gray explicitly asks for bedroom counts or a full listing recap, assess those requested facts too. "
        "The full reference below is available information, not an unconditional minimum for a narrower request. "
        "Do not add private finances, personal access histories or other participants' private needs. "
        "This does not waive any other participant's separate bedroom/accessibility goal. "
        "An absent recap question is not a supplied request and cannot prove an answered recap. "
        "Full public reference and privacy boundary: " + criteria['g_extra5']
    )
    return out
