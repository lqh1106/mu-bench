"""High-precision trigger for alleged missing INPUT provenance, not task absence."""
import re
REVISION='source-allegations-negative-items-r2-20261118'
ACCUSATION=re.compile(r'\b(?:invent(?:ed|s|ing)?|fabricat(?:e|ed|es|ing|ion)|hallucinat(?:e|ed|es|ing|ion)|made[ -]up|unsupported|unsubstantiated)\b|编造|虚构|凭空|无依据|没有依据',re.I)
ABSENCE=re.compile(r'\b(?:no\s+(?:supporting\s+)?(?:evidence|basis)|not\s+(?:provided|stated|established|specified|supplied)|(?:does|do|did)\s+not\s+(?:establish|specify|state|provide|support)|never\s+(?:mentioned|stated|provided))\b|未提供|未说明|未提及|没有表明|没有说明|缺乏证据',re.I)
INPUT=re.compile(r'\b(?:(?:supplied|provided|delivered|available|given|initial)\s+(?:input|context|information|material|sources?|facts?|notes?|documents?)|(?:initial\s+)?(?:background|context|registration|enrollment)|(?:user|participant)\s+(?:input|utterance|statement|speech|transcript)|source\s+(?:material|document|record)|input\s+(?:material|source)|case\s+(?:facts|context|input))\b|注册|初始资料|输入资料|已提供的背景|原始资料|用户发言',re.I)
BEHAVIOR_HEAD=re.compile(r'\b(?:(?:committed\s+)?assistant(?:.s)?\s+(?:speech|answer|response|behavior|recap|summary)|committed\s+(?:speech|answer|response))\b',re.I)
BEHAVIOR_TARGET=re.compile(r'\b(?:(?:assistant|committed)\s+(?:speech|response|answer|behavior)|(?:required|requested|final|spoken|completed)\s+(?:answer|outcome|summary|response|facts|result|recap)|completion|continuity|thread\s+preservation|speaker\s+anchor)\b|助手没有回答|未完成|没有完成',re.I)

def _clause_start(text,pos):
    return max([text.rfind(c,0,pos) for c in '.;!?\n'])+1

def is_negated_accusation(text,match):
    start=_clause_start(text,match.start());prefix=text[start:match.start()].casefold();word=match.group(0).casefold();suffix=text[match.end():match.end()+55].casefold()
    patterns=[r'\b(?:no|without)\s+(?:(?:any|new|additional|clear|unnecessary|demonstrated)\s+){0,3}$',
              r'\b(?:does|did|do|is|was|has|have|were|are)\s+not\s+(?:appear\s+to\s+)?$',
              r'\b(?:doesn.t|didn.t|isn.t|wasn.t|hasn.t)\s*$',
              r'\bnot\s+(?:(?:itself|in\s+itself)\s+)?(?:an?\s+)?$',
              r'\b(?:avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|absence\s+of)\s+(?:any\s+)?$',
              r'\bno\s+(?:evidence|sign|indication)\s+(?:of|for)\s*$',r'non[- ]$']
    if any(re.search(p,prefix) for p in patterns):return True
    if re.search(r'\bnot\s+to\s*$',prefix):return True
    if re.search(r'\bwithout[^.;]{0,100}\b(?:or|and)\s*$',prefix) and not re.search(r'\b(?:but|however|instead)\b',prefix[prefix.rfind('without'):]):return True
    if re.search(r'\b(?:does|did|do)\s+not\s+(?:make|add|introduce|assert|state|report|provide|include|give)\s+(?:(?:an?|any|new|additional)\s+){0,3}$',prefix):return True
    if re.match(r'\s+no\b',suffix):return True
    if re.match(r'^\s*no\b',prefix) and not re.search(r'\b(?:but|however|instead|and\s+(?:it|the\s+assistant))\b',prefix):return True
    if re.search(r'\b(?:cannot|does\s+not|did\s+not)\s+(?:establish|determine|show)[^.;]{0,65}\b(?:whether|if)\b',prefix):return True
    if word in ('invent','fabricate','hallucinate') and re.search(r'\b(?:does|did|do)\s+not[^.;]{0,100}(?:,\s*|\b(?:or|and)\s+)$',prefix):return True
    avoided=list(re.finditer(r'\bavoid(?:s|ed|ing)?\b',prefix))
    if avoided:
        governed=prefix[avoided[-1].end():]
        if len(governed)<90 and re.search(r'(?:,\s*|\b(?:or|and)\s+)$',governed) and not re.search(r'\b(?:but|however|instead)\b',governed):return True
    if word in ('invent','fabricate','hallucinate') and re.search(r'\b(?:does|did|do)\s+not(?:\s+[a-z]+){0,6}\s+(?:or|and)\s*$',prefix):return True
    if re.search(r'^\s*No\b.{0,100}\b(?:is|was|are|were)\b[^.;]*\b(?:or|and)\s*$',prefix,re.I):return True
    if re.match(r'\s+(?:is|was|are|were)\s+(?:not\s+(?:present|observed)|absent)',suffix):return True
    return False

def matches(reason):
    result=[]
    for m in ACCUSATION.finditer(reason):
        if not is_negated_accusation(reason,m):result.append({'kind':'explicit_factual_accusation','keyword':m.group(0),'span':[m.start(),m.end()]})
    for m in ABSENCE.finditer(reason):
        # Missing assistant action is not a claim that input facts were absent.
        prefix=reason[max(0,m.start()-260):m.start()];tail=reason[m.end():m.end()+150]
        inputs=list(INPUT.finditer(prefix));behaviors=list(BEHAVIOR_HEAD.finditer(prefix))
        before=bool(inputs and (not behaviors or inputs[-1].start()>behaviors[-1].start()))
        after=bool(re.match(r'\s+(?:in|by|from|among)\s+(?:the\s+)?',tail,re.I) and INPUT.search(tail[:110]))
        if (before or after) and not BEHAVIOR_TARGET.search(tail[:110]):
            result.append({'kind':'explicit_input_absence','keyword':m.group(0),'span':[m.start(),m.end()]})
    return result

def failed_item(iid,row,expected):
    if iid.startswith('goal:'):
        gid=iid.split(':',1)[1]
        return any(c.get('passed') is False for c in row.get('completion_checks',[])) or (gid in expected and row.get('disposition')!=expected[gid])
    if iid=='privacy:all_requirements':return bool(row.get('reason'))
    return row.get('passed') is False
