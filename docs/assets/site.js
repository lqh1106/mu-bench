'use strict';
(()=>{
 const dataset=JSON.parse(document.getElementById('leaderboard-data').textContent);
 const chart=document.getElementById('rank-chart'),rows=document.getElementById('chart-rows');
 const state={track:'tool_free',metric:0,family:'all',query:''};
 const metrics=['JTS','Attribution','Threading','Coordination'];
 let visible=false;
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function filtered(){return dataset.models.filter(m=>m[state.track]&&(state.family==='all'||m.family===state.family)&&m.name.toLowerCase().includes(state.query)).sort((a,b)=>b[state.track][state.metric]-a[state.track][state.metric]);}
 function render(){
  chart.classList.remove('is-expanded');const models=filtered();
  rows.innerHTML=models.length?models.map((m,i)=>{const value=m[state.track][state.metric];return `<div class="chart-row" role="listitem" aria-label="${esc(m.name)}: ${metrics[state.metric]} ${value.toFixed(2)} percent"><span class="chart-model" title="${esc(m.name)}">${esc(m.name)}</span><div class="bar-slot" style="--score:${value}%;--delay:${Math.min(i*35,350)}ms"><span class="bar-fill ${m.family}" title="${metrics[state.metric]}: ${value.toFixed(2)}%"></span><span class="bar-value">${value.toFixed(2)}</span></div></div>`;}).join(''):'<p class="chart-empty">No evaluated models match your selection.</p>';
  document.getElementById('result-count').textContent=`${models.length} evaluated ${models.length===1?'system':'systems'} · scores in %`;
  document.getElementById('chart-caption').textContent=`${metrics[state.metric]==='JTS'?'Joint Task Success':metrics[state.metric]} (%)`;
  document.getElementById('chart-note').textContent=state.track==='tool_augmented'?'Paper Table 2 snapshot. Four systems have Tool-Augmented results; unevaluated systems are omitted. Higher is better.':'Paper Table 2 snapshot. Scores are averaged over evaluated cases; higher is better. JTS is the primary outcome; A, T and C are diagnostic views.';
  if(visible)requestAnimationFrame(()=>requestAnimationFrame(()=>chart.classList.add('is-expanded')));
 }
 document.querySelectorAll('[data-chart-track]').forEach(b=>b.addEventListener('click',()=>{state.track=b.dataset.chartTrack;document.querySelectorAll('[data-chart-track]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();}));
 document.getElementById('chart-metric').addEventListener('change',e=>{state.metric=Number(e.target.value);render();});
 document.getElementById('chart-family').addEventListener('change',e=>{state.family=e.target.value;render();});
 document.getElementById('chart-search').addEventListener('input',e=>{state.query=e.target.value.trim().toLowerCase();render();});
 document.getElementById('download').addEventListener('click',()=>{const columns=['System','Family','TF_JTS','TF_A','TF_T','TF_C','TA_JTS','TA_A','TA_T','TA_C'];const data=[columns,...dataset.models.map(m=>[m.name,m.family,...m.tool_free,...(m.tool_augmented??['','','',''])])];const csv=data.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');const a=document.createElement('a');const url=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='mu-bench-scores.csv';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 render();
 if('IntersectionObserver'in window)new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){visible=true;chart.classList.add('is-expanded');}},{threshold:.18}).observe(chart);else{visible=true;chart.classList.add('is-expanded');}
})();
document.getElementById('copy').addEventListener('click',async()=>{const b=document.getElementById('copy'),s=document.getElementById('copy-status');try{await navigator.clipboard.writeText(document.getElementById('quickstart').textContent);b.textContent='Copied';s.textContent='Quick-start commands copied.';setTimeout(()=>b.textContent='Copy',2000);}catch{s.textContent='Select the commands to copy them manually.';}});
