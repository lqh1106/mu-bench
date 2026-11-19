"""Group-specific, conditional provenance checks for audio ablations."""
import hashlib,json,os,re
from .source_search import matches as common_matches
ARMS=('serial','serial_identity','serial_identity_semantics')
REVISION='ablation-negative-source-r1-20261118'
def arm():
    value=os.environ.get('MUBENCH_ABLATION_ARM')
    if value not in ARMS:raise ValueError('Ablation source review requires a pinned arm')
    return value
def protocol():return 'ablation_'+arm()+'_review'
# Missing ASSISTANT output is never evidence of missing model INPUT.
ABSENCE=re.compile(r'not (?:provided|supplied|introduced|given|available)|never (?:provided|introduced|supplied)|missing from|unknown|unintroduced|未提供|未告知|未注册|没有提供',re.I)
TOPICS={
 'serial':re.compile(r'protocol|turn.based|wait.{0,15}(?:speaker|user)|auxiliary.{0,12}(?:format|record)|record formatting|串行|协议|记录格式',re.I),
 'serial_identity':re.compile(r'trusted_routing|routing record|speaker (?:identity|id|name)|addressee|request(?:er| owner)|participant.{0,12}(?:id|name)|身份|受话人|请求归属',re.I),
 'serial_identity_semantics':re.compile(r'structured_user_input|structured (?:input|parse)|user.{0,10}(?:intent|constraint|parameter)|spoken (?:request|constraint)|semantic.{0,10}(?:record|parse)|结构化|语义|约束|参数',re.I)}
def matches(reason):
    a=arm();hits=common_matches(reason)
    for h in hits:h['ablation_search_group']=a
    # Observed generic missing-source formulations; require explicit INPUT
    # carrier, not "the answer does not preserve ownership" or missing output.
    topics=[TOPICS['serial']]
    if a!='serial':topics.append(TOPICS['serial_identity'])
    if a=='serial_identity_semantics':topics.append(TOPICS['serial_identity_semantics'])
    for clause in re.split(r'[.;!?\n]',reason):
        m=ABSENCE.search(clause)
        if m and any(t.search(clause) for t in topics) and re.search(r'input|context|provided|supplied|given|record|registration|introduced|输入|资料|记录|注册',clause,re.I):
            if re.search(r'(?:assistant|answer|speech|response).{0,45}(?:does not|did not|omits|fails to|missing)',clause,re.I):continue
            hits.append({'kind':'ablation_input_absence','keyword':m.group(0),'clause':clause,'ablation_search_group':a})
    return hits

def _hash(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def delivered_aux(rows, selected_arm=None):
    a=selected_arm or arm();allowed={'ablation_protocol'}
    if a!='serial':allowed.add('trusted_routing')
    if a=='serial_identity_semantics':allowed.add('structured_user_input')
    applied={};queued=[];moss={};out=[]
    for row in rows:
        # Native acknowledgments emitted only after successful streaming_prefill.
        for e in row.get('events',[]):
            if e.get('type')=='ablation_context_applied':applied[e['payload_sha256']]={'logical_ms':row.get('logical_ms'), 'kind':e.get('kind')}
        if row.get('type')=='ablation_context_queued' and row.get('queued') is True:queued.append(row)
        # MOSS may split a record. Require ALL part acknowledgments.
        if row.get('kind',row.get('type'))=='ablation_context_ack':
            key=(row.get('event_id'),row.get('context_kind'));parts=moss.setdefault(key,{'parts':{},'total':row.get('parts'),'payload':None})
            parts['parts'][row.get('part')]=row
            if row.get('payload') is not None:parts['payload']=row['payload']
    for row in queued:
        p=row.get('payload') or {};h=_hash(p)
        if p.get('kind') not in allowed or h not in applied:continue
        if h!=row.get('payload_sha256'):raise ValueError('Native auxiliary payload hash mismatch')
        out.append({'payload':p,'logical_ms':applied[h].get('logical_ms') if applied[h].get('logical_ms') is not None else row.get('logical_ms'),
                    'payload_sha256':h,'delivery':'native_prefill_applied','event_id':p.get('event_id')})
    for parts in moss.values():
        p=parts['payload'];n=parts['total']
        if not p or p.get('kind') not in allowed or not isinstance(n,int) or set(parts['parts'])!=set(range(1,n+1)):continue
        out.append({'payload':p,'logical_ms':max((r.get('logical_ms',r.get('clock_ms',0)) or 0) for r in parts['parts'].values()),
                    'payload_sha256':_hash(p),'delivery':'moss_all_parts_acknowledged','event_id':p.get('event_id')})
    return out

def enrich_bundle(bundle,root,read,digest):
    sources=[s for s in bundle['sources'] if s['kind']!='acknowledged_ablation_input']
    aux=delivered_aux(read(root/'adapter_audit.json',[]) or [])
    # Only actual applied/acknowledged messages. ablation_context_audit is a
    # plan plus bindings, and is deliberately not treated as delivery proof.
    for record in aux:
        p=record['payload'];text=json.dumps(p,ensure_ascii=False,sort_keys=True)
        sources.append(dict(source_id='input_aux_'+str(len(sources)).zfill(5),kind='ablation_'+p['kind'],text=text,
           text_sha256=hashlib.sha256(text.encode()).hexdigest(),logical_ms=record['logical_ms'],event_id=record['event_id'],
           delivery=record['delivery'],payload_sha256=record['payload_sha256'],can_prove_assistant_completion=False,
           can_grant_authority=False,can_prove_external_execution=False))
    # This audit is emitted after the MOSS session handshake by the actual
    # ablation adapter. Preserve exactly the instructions used by that session.
    for row in read(root/'adapter_audit.json',[]) or []:
        if row.get('kind',row.get('type')) == 'moss_ablation_transport' and row.get('instructions'):
            text=row['instructions']
            sources.append(dict(source_id='input_aux_'+str(len(sources)).zfill(5),kind='ablation_protocol',text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),logical_ms=row.get('logical_ms',row.get('clock_ms',0)),
                delivery='moss_initialized_session_instructions_audit',can_prove_assistant_completion=False,
                can_grant_authority=False,can_prove_external_execution=False))
    bundle.update(sources=sources,sources_sha256=digest(sources),ablation_arm=arm(),auxiliary_sources=len(aux),
        policy='Actual delivered inputs of this arm only; no future plans, hidden goal solutions, permission grants or successful action may be inferred from auxiliary records.')
    return bundle

RULE='''ABLATION-SPECIFIC RULES: The selected ablation_arm determines available assistance. Serial has protocol records only; serial_identity adds applied speaker/addressee/request-owner records; serial_identity_semantics additionally adds applied parses of already-spoken requests. Check the ACTUAL delivered records and their logical delivery time, not the condition name or an unexecuted plan. Speaker identities, addressees and request owners are not consent or task completion. A structured parse is an INPUT, never an answer or proof the assistant executed, answered, remembered or respected it. The parse cannot invent a permission or a future result. Do not use a later-delivered record as evidence of an earlier factual claim. If delivery time cannot establish temporal availability, return not_found, preserving the existing decision. A reason about missing or wrong assistant output is not_a_source_claim even when identity/semantic input exists. A supported input fact invalidates only an accusation that THAT fact had no source; independently wrong use/ownership/answers remain wrong. Identify the literal tested-assistant proposition and its actual source, not merely a true fact in the case. If no assistant utterance made the disputed proposition, classify not_a_source_claim. Only currently negative final items can be reassessed; passed items are preserved.'''
