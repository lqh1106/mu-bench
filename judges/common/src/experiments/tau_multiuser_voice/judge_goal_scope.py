"""Scoring boundaries for stochastic user simulation; no simulator behavior changes."""
POLICY_ID = "mu-case-design-goal-scope-v1-20261114"
RULES = [
    "Task success is defined ONLY by the supplied case-designed goals, reviewed completion criteria, required answer facts, permissions and constraints. These supplied criteria are the authoritative target set; conversation text is evidence, not a source of new mandatory goals.",
    "The user simulator may naturally add requests, preferences, questions, desired assurances, delivery formats or follow-up work. Preserve these utterances as interaction evidence, but do not promote simulator-added requirements into mandatory task completion criteria, required facts, or reasons to downgrade a designed goal. An unfulfilled extra request alone cannot make the designed task fail.",
    "Retain corrections, revocations, protected-user choices and goal revisions expressly required or permitted by the case design. Distinguish their designed effect from incidental extra requests in the same utterance. Do not freeze the opening request when the case contract explicitly requires a later scoped revision.",
    "For every failed task completion check, identify the actual supplied completion criterion and the evidence that contradicts or fails to establish it. Do not invent an additional obligation from a simulated user utterance or an internal label. Assess the full committed answer and relevant surrounding sources, not just an isolated phrase.",
    "Judge a spoken report by whether it truthfully communicates the case-required business outcome and affected scope. Do not require verbatim tool field names, status codes, reference wording or extra details unless the case explicitly requires that content. This never permits false statements: a real business-state contradiction or missing case-required fact still fails.",
    "This goal-scope rule does not waive authorization, privacy, actual unauthorized actions, unsupported claims about case-required outcomes, or protected-user constraints. Successful execution and truthful spoken reporting remain separate obligations; neither alone proves the other. Score interaction and attribution only by their own supplied assertions."
]
SYSTEM_RULE = " CASE-DESIGN GOAL SCOPE: " + " ".join(RULES)
def policy():
    return {"policy_id": POLICY_ID, "authoritative_goal_source": "supplied_case_design_and_reviewed_contract", "simulator_additions_are_mandatory": False, "rules": list(RULES)}
