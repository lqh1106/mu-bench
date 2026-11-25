'use strict';
(async()=>{
 const table=document.getElementById('results'),count=document.getElementById('result-count');let dataset;
 try{const res=await fetch('assets/leaderboard.json');if(!res.ok)throw new Error('Data unavailable');dataset=await res.json();}
 catch{table.querySelector('tbody').innerHTML='<tr><td class="empty">Leaderboard data could not be loaded. Please reload the page.</td></tr>';return;}
 const state={track:'both',family:'all',query:'',sortTrack:'tool_free',sortMetric:0,descending:true};
 const metrics=['JTS','A','T','C'],titles=['Joint Task Success','Attribution','Threading','Coordination'];
 const familyNames={open:'Open-weight speech models',proprietary:'Proprietary voice agents'};
 const tracks=()=>state.track==='both'?['tool_free','tool_augmented']:[state.track];
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const filtered=()=>dataset.models.filter(m=>(state.family==='all'||m.family===state.family)&&m.name.toLowerCase().includes(state.query));
 const score=m=>m[state.sortTrack]?.[state.sortMetric]??null;
 function sorted(rows){return rows.slice().sort((a,b)=>{const x=score(a),y=score(b);if(x===null)return y===null?0:1;if(y===null)return-1;return state.descending?y-x:x-y;});}
 function render(){
  const visible=tracks(),models=filtered();
  table.querySelector('thead').innerHTML='<tr><th class="model-heading" rowspan="2" scope="col">System</th>'+visible.map(t=>`<th colspan="4" scope="colgroup" class="${t==='tool_augmented'?'track-separator':''}">${t==='tool_free'?'Tool-Free':'Tool-Augmented'}</th>`).join('')+'</tr><tr>'+visible.map(t=>metrics.map((m,i)=>{const active=t===state.sortTrack&&i===state.sortMetric;return `<th scope="col" aria-sort="${active?(state.descending?'descending':'ascending'):'none'}" class="${i===0?'jts ':''}${t==='tool_augmented'&&i===0?'track-separator':''}"><button class="metric-sort ${active?'active':''}" type="button" data-sort-track="${t}" data-metric="${i}" title="Sort by ${titles[i]} on ${t==='tool_free'?'Tool-Free':'Tool-Augmented'}">${m}<span class="sort-mark" aria-hidden="true">${active?(state.descending?'▾':'▴'):'↕'}</span></button></th>`;}).join('')).join('')+'</tr>';
  const body=[];
  for(const family of ['open','proprietary']){
   const rows=sorted(models.filter(m=>m.family===family));if(!rows.length)continue;
   body.push(`<tr class="family-row"><th colspan="${1+visible.length*4}" scope="colgroup">${familyNames[family]}</th></tr>`);
   for(const m of rows){body.push('<tr class="data-row"><th class="model-cell" scope="row">'+esc(m.name)+'</th>'+visible.map(t=>metrics.map((_,i)=>{const v=m[t]?.[i];const eligible=dataset.models.filter(x=>t==='tool_augmented'||x.family===family).map(x=>x[t]?.[0]).filter(x=>x!==undefined);const best=i===0&&v!==undefined&&v===Math.max(...eligible);return `<td class="${i===0?'jts ':''}${t==='tool_augmented'&&i===0?'track-separator ':''}${v===undefined?'missing ':''}${best?'best':''}">${v===undefined?'<span aria-label="Not evaluated">—</span>':v.toFixed(2)}</td>`;}).join('')).join('')+'</tr>');}
  }
  table.querySelector('tbody').innerHTML=body.length?body.join(''):`<tr><td colspan="${1+visible.length*4}" class="empty">No models match your search.</td></tr>`;
  count.textContent=`${models.length} ${models.length===1?'system':'systems'} · scores in %`;
 }
 document.querySelectorAll('[data-track]').forEach(b=>b.addEventListener('click',()=>{state.track=b.dataset.track;if(state.track!=='both')state.sortTrack=state.track;document.querySelectorAll('[data-track]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();}));
 document.getElementById('search').addEventListener('input',e=>{state.query=e.target.value.toLowerCase().trim();render();});
 document.getElementById('family').addEventListener('change',e=>{state.family=e.target.value;render();});
 table.addEventListener('click',e=>{const b=e.target.closest('[data-sort-track]');if(!b)return;const i=Number(b.dataset.metric);state.descending=b.dataset.sortTrack===state.sortTrack&&i===state.sortMetric?!state.descending:true;state.sortTrack=b.dataset.sortTrack;state.sortMetric=i;render();});
 document.getElementById('download').addEventListener('click',()=>{const visible=tracks(),rows=[['System','Family',...visible.flatMap(t=>metrics.map(m=>`${t}_${m}`))],...['open','proprietary'].flatMap(f=>sorted(filtered().filter(m=>m.family===f))).map(m=>[m.name,familyNames[m.family],...visible.flatMap(t=>metrics.map((_,i)=>m[t]?.[i]?.toFixed(2)??''))])];const csv=rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');const url=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='mu-bench-table2-'+state.track+'.csv';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 render();
})();
document.getElementById('copy').addEventListener('click',async()=>{const b=document.getElementById('copy'),s=document.getElementById('copy-status');try{await navigator.clipboard.writeText(document.getElementById('quickstart').textContent);b.textContent='Copied';s.textContent='Quick-start commands copied.';setTimeout(()=>b.textContent='Copy',2000);}catch{s.textContent='Copy unavailable. Select the commands to copy them manually.';b.textContent='Select text';}});
