'use strict';
(()=>{
 const demo=document.getElementById('demo');if(!demo)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const progress=document.getElementById('demo-progress'),play=document.getElementById('demo-play');
 const canvas=document.getElementById('timeline-canvas'),viewport=document.getElementById('timeline-viewport'),axis=document.getElementById('time-axis');
 const steps=[
  {at:.025,end:.17,left:1,width:19,who:'dad',tag:'To agent',text:'Get us to the airport by 10.',axis:'A',title:'The shared task begins.',insight:'Dad sets the destination and deadline. They remain in force throughout the conversation.',eta:'Not checked',approval:'Not requested',result:'Main task received'},
  {at:.17,end:.32,left:17,width:17,who:'agent',tag:'Planning the route',text:'The fast route to…',tool:'route_planner()',axis:'T',title:'The agent starts the main thread.',insight:'The route is being planned when another participant speaks.',eta:'Not checked',approval:'Not requested',result:'Route planned'},
  {at:.24,end:.4,left:24,width:21,who:'child',tag:'Barge-in · to agent',text:'Can we stop by the park?',axis:'A',title:'A new voice interrupts the reply.',insight:'The child addresses the agent mid-response. The park request must be considered alongside the airport deadline.',eta:'Not checked',approval:'Awaiting approval',result:'New request · original goal retained'},
  {at:.4,end:.61,left:40,width:23,who:'agent',tag:'Checking the detour',text:'We’d still reach the airport by 9:52. Should I add the stop?',tool:'check_eta() → 09:52',axis:'C',title:'Check feasibility before changing the plan.',insight:'The ETA fits the deadline. The agent asks for approval before adding the stop.',eta:'09:52',approval:'Awaiting approval',result:'09:52 < 10:00'},
  {at:.61,end:.8,left:61,width:20,who:'dad',tag:'Approval · deadline retained',text:'Yes, as long as we arrive by 10.',axis:'C',title:'Approval carries a constraint.',insight:'Dad approves the stop and keeps the original arrival deadline.',eta:'09:52',approval:'Approved by Dad',result:'Deadline retained'},
  {at:.69,end:.8,left:69,width:17,who:'child',tag:'Side-talk · to Dad',text:'Yay! Thanks, Dad!',axis:'A',title:'Side-talk is not a new agent request.',insight:'The child is thanking Dad. The pending route update should continue.',eta:'09:52',approval:'Approved by Dad',result:'No new agent request'},
  {at:.8,end:1,left:80,width:19,who:'agent',tag:'Main task resumed',text:'We’ll stop at the park first… then head to the airport.',tool:'route_planner(via park)',axis:'T',title:'Resume the task with the approved update.',insight:'The route now includes the park, while the airport and deadline remain attached to the main thread.',eta:'09:52',approval:'Approved by Dad',result:'Route updated'},
  {at:.89,end:.97,left:88,width:11,who:'dad',tag:'Backchannel',text:'Mm-hmm.',axis:'C',title:'An acknowledgment leaves the plan intact.',insight:'The park stop is approved and the ETA remains before 10. This brief acknowledgment introduces no new task.',eta:'09:52',approval:'Approved by Dad',result:'Constraints satisfied'}
 ];
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 steps.forEach((s,i)=>{
  const el=document.createElement('article');el.className=`timeline-event event-${s.who}`;el.dataset.event=i;el.style.left=`${s.at*100}%`;el.style.width=`${s.width}%`;el.setAttribute('aria-hidden','true');
  el.innerHTML=`<span class="event-tag">${s.tag}</span>${s.tool?`<code>${esc(s.tool)}</code>`:''}<p>“${esc(s.text)}”</p>${i===1?'<span class="interrupted-tag">Interrupted</span>':''}`;
  document.getElementById(`lane-${s.who}`).append(el);
  const strip=document.createElement('span');strip.className=`audio-segment segment-${s.who}`;strip.style.left=`${s.at*100}%`;strip.style.width=`${(s.end-s.at)*100}%`;document.getElementById('audio-strips').append(strip);
 });
 // Speaker lanes reorder the DOM; bind each bubble to its chronological event explicitly.
 const events=steps.map((step,eventId)=>({step,el:demo.querySelector(`[data-event="${eventId}"]`)}));
 let fraction=0,index=-2,playing=false,frame=null,lastTime=0,autoConsumed=false,started=false;
 function controls(){demo.dataset.playing=String(playing);document.getElementById('play-label').textContent=playing?'Pause':fraction>=1?'Replay':'Play';document.getElementById('play-symbol').textContent=playing?'Ⅱ':'▶';play.setAttribute('aria-label',playing?'Pause walkthrough':fraction>=1?'Replay walkthrough':'Play walkthrough');document.getElementById('demo-status').textContent=playing?'Playing illustration':fraction>=1?'Walkthrough complete':started?'Paused':'Ready to play';}
 function pan(){const x=axis.offsetLeft+axis.clientWidth*fraction;viewport.scrollLeft=Math.max(0,x-viewport.clientWidth*.72);}
 function render(){
  const next=steps.reduce((n,s,i)=>fraction>=s.at?i:n,-1);
  demo.dataset.phase=String(next);const cursor=axis.offsetLeft+axis.clientWidth*fraction;document.getElementById('time-cursor').style.left=`${cursor}px`;document.getElementById('time-elapsed').style.width=`${fraction*100}%`;document.getElementById('audio-strips').style.clipPath=`inset(0 ${(1-fraction)*100}% 0 0)`;document.getElementById('retained-progress').style.width=`${fraction*100}%`;
  document.getElementById('axis-handle').style.left=`${fraction*100}%`;axis.setAttribute('aria-valuenow',String(Math.round(fraction*1000)/10));
  const seconds=Math.min(40,Math.floor(fraction*40));axis.setAttribute('aria-valuetext',`00:${String(seconds).padStart(2,'0')} of 00:40`);document.getElementById('timeline-time').innerHTML=`00:${String(seconds).padStart(2,'0')} <span>/ 00:40</span>`;progress.value=String(fraction*100);progress.style.setProperty('--progress',`${fraction*100}%`);
  events.forEach(({el,step})=>{const delivered=fraction>=step.at;el.classList.toggle('delivered',delivered);el.classList.toggle('speaking',delivered&&fraction<step.end);el.setAttribute('aria-hidden',String(!delivered));});
  document.querySelector('.interrupted-tag').classList.toggle('shown',fraction>=steps[2].at);
  document.getElementById('thread-status').textContent=fraction>=.8?'Retained through the detour':'Main thread';
  if(next!==index){index=next;const s=steps[Math.max(0,index)];
   for(const [id,value] of Object.entries({'state-eta':s.eta,'state-approval':s.approval,'tool-result':index<0?'Waiting for a request':s.result,'insight-title':index<0?'Follow the shared audio stream.':s.title,'insight-text':index<0?'Press play. Utterances unfold from left to right, including interruptions, side-talk and brief acknowledgments.':s.insight,'step-count':`${String(index+1).padStart(2,'0')} / 08`}))document.getElementById(id).textContent=value;
   const axis=document.getElementById('insight-axis');axis.textContent=index<0?'μ':s.axis;axis.dataset.axis=s.axis;
   document.getElementById('state-eta').classList.toggle('positive',index>=3);document.getElementById('state-approval').classList.toggle('positive',index>=4);
   progress.setAttribute('aria-valuetext',index<0?'Ready to play':`Step ${index+1}: ${s.title}`);
  }
 }
 function tick(now){if(!playing)return;fraction=Math.min(1,fraction+(now-lastTime)/40000);lastTime=now;render();pan();if(fraction>=1){pause();return;}frame=requestAnimationFrame(tick);}
 function start(){started=true;if(fraction>=1)fraction=0;playing=true;lastTime=performance.now();controls();frame=requestAnimationFrame(tick);}
 function pause(){playing=false;cancelAnimationFrame(frame);frame=null;controls();}
 play.addEventListener('click',()=>{autoConsumed=true;playing?pause():start();});
 document.getElementById('demo-replay').addEventListener('click',()=>{autoConsumed=true;pause();fraction=0;render();pan();start();});
 function seek(value,follow=true){autoConsumed=true;started=true;pause();fraction=Math.max(0,Math.min(1,value));render();if(follow)pan();controls();}
 progress.addEventListener('input',e=>seek(Number(e.target.value)/100));
 let dragPointer=null;
 function seekPointer(e){const bounds=axis.getBoundingClientRect();seek((e.clientX-bounds.left)/bounds.width,false);}
 axis.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragPointer=e.pointerId;axis.setPointerCapture(e.pointerId);axis.classList.add('dragging');seekPointer(e);e.preventDefault();});
 axis.addEventListener('pointermove',e=>{if(e.pointerId===dragPointer)seekPointer(e);});
 function endDrag(e){if(e.pointerId!==dragPointer)return;dragPointer=null;axis.classList.remove('dragging');if(axis.hasPointerCapture(e.pointerId))axis.releasePointerCapture(e.pointerId);}
 axis.addEventListener('pointerup',endDrag);axis.addEventListener('pointercancel',endDrag);axis.addEventListener('lostpointercapture',()=>{dragPointer=null;axis.classList.remove('dragging');});
 axis.addEventListener('keydown',e=>{let target;switch(e.key){case 'ArrowLeft':case 'ArrowDown':target=fraction-.025;break;case 'ArrowRight':case 'ArrowUp':target=fraction+.025;break;case 'PageDown':target=fraction-.1;break;case 'PageUp':target=fraction+.1;break;case 'Home':target=0;break;case 'End':target=1;break;default:return;}e.preventDefault();seek(target);});
 document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=null;}else if(playing){lastTime=performance.now();frame=requestAnimationFrame(tick);}});reduced.addEventListener('change',()=>{if(reduced.matches)pause();});
 if('ResizeObserver' in window)new ResizeObserver(()=>{render();if(playing)pan();}).observe(canvas);
 render();controls();
 if('IntersectionObserver' in window){const auto=new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting&&!autoConsumed&&!reduced.matches){autoConsumed=true;start();}}},{threshold:.25});auto.observe(demo);
 const reveal=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){e.target.classList.add('visible');reveal.unobserve(e.target);}},{threshold:.08});document.querySelectorAll('.overview .section-heading,.track-grid,.framework .section-heading,.metric-explainer,#leaderboard .section-heading,.board,.start-grid').forEach(el=>{el.classList.add('reveal');reveal.observe(el);});}
})();
