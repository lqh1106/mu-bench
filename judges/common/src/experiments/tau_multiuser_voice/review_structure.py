"""Project a partial accepted review onto authored items, without inventing verdicts."""
MISSING = '__judge_item_not_returned__'

def manifest(contract, packet):
    items = []
    for goal in contract['goals']:
        gid = goal['goal_id']
        items.append(('goal:' + gid, 'goal', gid, None))
        for i, criterion in enumerate(goal.get('completion_criteria', [])):
            items.append((f'completion:{gid}:{i}', 'completion', gid, criterion))
        for fact in goal.get('required_answer_facts', []):
            items.append((f"fact:{gid}:{fact['id']}", 'fact', gid, fact['id']))
    for assertion in packet['assertions']:
        items.append(('capability:' + assertion['assertion_id'], 'capability', assertion['assertion_id'], None))
    items.append(('privacy:all_requirements', 'privacy', None, None))
    if len({i[0] for i in items}) != len(items):
        raise ValueError('Authored review manifest has duplicate IDs')
    return items

def _rows(payload, key):
    rows = payload.get(key)
    return rows if isinstance(rows, list) else []

def project(payload, items):
    """Missing stays unknown. No empty list, False, or success is substituted."""
    import json
    from .judge_evidence import normalize_quote
    values = {}
    for iid, kind, owner, detail in items:
        value = MISSING
        if kind in ('goal', 'completion'):
            matches = [r for r in _rows(payload, 'goal_results') if isinstance(r, dict) and r.get('goal_id') == owner]
            row = matches[0] if len(matches) == 1 else {}
            if kind == 'goal':
                if isinstance(row.get('disposition'), str):value = row['disposition']
            else:
                checks = [r for r in _rows(row, 'completion_checks') if isinstance(r, dict) and normalize_quote(str(r.get('criterion') or '')) == normalize_quote(detail)]
                if len(checks) == 1 and type(checks[0].get('passed')) is bool:
                    value = (checks[0]['passed'], checks[0].get('applicability', 'applicable'))
        elif kind in ('fact', 'capability'):
            key = 'fact_checks' if kind == 'fact' else 'capability_results'
            rows = [r for r in _rows(payload, key) if isinstance(r, dict) and
                    (r.get('goal_id') == owner and r.get('id') == detail if kind == 'fact' else r.get('assertion_id') == owner)]
            if len(rows) == 1 and type(rows[0].get('passed')) is bool:value = rows[0]['passed']
        elif kind == 'privacy' and isinstance(payload.get('privacy_leaks'), list):
            value = json.dumps(payload['privacy_leaks'], sort_keys=True, ensure_ascii=False)
        values[iid] = value
    return values, [iid for iid, value in values.items() if value == MISSING]
