"""Aggregate only sealed results; keep nonapplicable RICE values absent."""
from pathlib import Path
import argparse,json,csv,collections
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups=collections.defaultdict(list)
for receipt in a.root.rglob('release_receipt.json'):
 d=receipt.parent
 if 'formal' not in d.parts:continue
 score=json.loads((d/'score.json').read_text());rice=json.loads((d/'rice_score.json').read_text());identity=d.name.split('__');condition=tuple(identity[1:]);relative=d.relative_to(a.root).parts
 model=relative[0];subset=relative[1];arm=relative[2]
 for variable in [('overall',),condition]:groups[(model,subset,arm,*variable)].append((score,rice))
rows=[]
for key,values in sorted(groups.items()):
 row={'model':key[0],'subset':key[1],'condition':key[2],'variant':':'.join(key[3:]),'task_success_pct':100*sum(s['task_success'] for s,r in values)/len(values),'authorization_success_pct':100*sum(s.get('authorization_success',True) for s,r in values)/len(values)}
 for axis in ['routing','continuity','isolation','equity']:
  vs=[r[axis+'_score'] for s,r in values if r.get(axis+'_score') is not None];row[axis+'_pct']=100*sum(vs)/len(vs) if vs else None
 rows.append(row)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['model','subset','condition','variant','task_success_pct','authorization_success_pct','routing_pct','continuity_pct','isolation_pct','equity_pct']);writer.writeheader();writer.writerows(rows)
print(json.dumps({'output':str(a.output),'groups':len(rows)}))
