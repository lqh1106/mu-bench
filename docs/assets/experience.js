'use strict';
(()=>{
 const demo=document.getElementById('demo');if(!demo)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const durationSeconds=18;
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
  {at:.89,end:1,left:88,width:11,who:'dad',tag:'Backchannel',text:'Mm-hmm.',axis:'C',title:'An acknowledgment leaves the plan intact.',insight:'The park stop is approved and the ETA remains before 10. This brief acknowledgment introduces no new task.',eta:'09:52',approval:'Approved by Dad',result:'Constraints satisfied'}
 ];
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 steps.forEach((s,i)=>{
  const el=document.createElement('article');el.className=`timeline-event event-${s.who}`;el.dataset.event=i;el.style.left=`${s.at*100}%`;el.style.width=`${(s.end-s.at)*100}%`;el.setAttribute('aria-hidden','true');
  el.innerHTML=`<span class="event-tag">${s.tag}</span>${s.tool?`<code>${esc(s.tool)}</code>`:''}<p class="spoken-text" aria-label=""><span aria-hidden="true">“</span>${s.text.split(' ').map(word=>`<span class="spoken-word" aria-hidden="true">${esc(word)} </span>`).join('')}<span class="closing-quote" aria-hidden="true">”</span></p>${i===1?'<span class="interrupted-tag">Interrupted</span>':''}`;
  document.getElementById(`lane-${s.who}`).append(el);
  const strip=document.createElement('span');strip.className=`audio-segment segment-${s.who}`;strip.style.left=`${s.at*100}%`;strip.style.width=`${(s.end-s.at)*100}%`;document.getElementById('audio-strips').append(strip);
 });
 // Speaker lanes reorder the DOM; bind each bubble to its chronological event explicitly.
 const events=steps.map((step,eventId)=>{const el=demo.querySelector(`[data-event="${eventId}"]`);return {step,el,words:Array.from(el.querySelectorAll('.spoken-word')),wordCount:-1};});
 let fraction=0,index=-2,playing=false,frame=null,lastTime=0,autoConsumed=false,started=false,scrubbing=false,dragPointer=null,loopHoldUntil=0,loopCount=0;
 function controls(){demo.dataset.playing=String(playing);demo.dataset.scrubbing=String(scrubbing);document.getElementById('play-label').textContent=playing?'Pause':fraction>=1?'Replay':'Play';document.getElementById('play-symbol').textContent=playing?'Ⅱ':'▶';play.setAttribute('aria-label',playing?'Pause walkthrough':fraction>=1?'Replay walkthrough':'Play walkthrough');document.getElementById('demo-status').textContent=scrubbing?'Seeking…':playing?(fraction>=1?'Loop complete · restarting':'Playing illustration'):fraction>=1?'Walkthrough complete':started?'Paused':'Ready to play';}
 function pan(){const x=axis.offsetLeft+axis.clientWidth*fraction;viewport.scrollLeft=Math.max(0,x-viewport.clientWidth*.72);}
 function render(){
  const next=steps.reduce((n,s,i)=>fraction>=s.at?i:n,-1);
  demo.dataset.phase=String(next);const cursor=axis.offsetLeft+axis.clientWidth*fraction;document.getElementById('time-cursor').style.left=`${cursor}px`;document.getElementById('time-elapsed').style.width=`${fraction*100}%`;document.getElementById('audio-strips').style.clipPath=`inset(0 ${(1-fraction)*100}% 0 0)`;document.getElementById('retained-progress').style.width=`${fraction*100}%`;
  document.getElementById('axis-handle').style.left=`${fraction*100}%`;axis.setAttribute('aria-valuenow',String(Math.round(fraction*1000)/10));
  const seconds=Math.min(durationSeconds,Math.floor(fraction*durationSeconds));axis.setAttribute('aria-valuetext',`00:${String(seconds).padStart(2,'0')} of 00:18`);document.getElementById('timeline-time').innerHTML=`00:${String(seconds).padStart(2,'0')} <span>/ 00:18</span>`;progress.value=String(fraction*100);progress.style.setProperty('--progress',`${fraction*100}%`);
  events.forEach(event=>{const {el,step,words}=event;const delivered=fraction>=step.at;const active=delivered&&fraction<step.end;el.classList.toggle('delivered',delivered);el.classList.toggle('speaking',active);el.setAttribute('aria-hidden',String(!delivered));
   const spoken=Math.max(0,Math.min(1,(fraction-step.at)/(step.end-step.at)));
   const wordCount=delivered?Math.min(words.length,Math.max(1,Math.ceil(spoken*words.length))):0;
   if(wordCount!==event.wordCount){event.wordCount=wordCount;words.forEach((word,i)=>{word.classList.toggle('revealed',i<wordCount);word.classList.toggle('word-current',i===wordCount-1&&active);});el.querySelector('.spoken-text').setAttribute('aria-label',step.text.split(' ').slice(0,wordCount).join(' '));}
   el.querySelector('.closing-quote').classList.toggle('revealed',spoken>=1);
  });
  for(const person of ['dad','child','agent']){const active=steps.some(step=>step.who===person&&fraction>=step.at&&fraction<step.end);demo.querySelector(`.lane-${person}`).classList.toggle('lane-speaking',active);}

  document.querySelector('.interrupted-tag').classList.toggle('shown',fraction>=steps[2].at);
  document.getElementById('thread-status').textContent=fraction>=.8?'Retained through the detour':'Main thread';
  if(next!==index){index=next;const s=steps[Math.max(0,index)];
   for(const [id,value] of Object.entries({'state-eta':s.eta,'state-approval':s.approval,'tool-result':index<0?'Waiting for a request':s.result,'insight-title':index<0?'Follow the shared audio stream.':s.title,'insight-text':index<0?'Press play. Utterances unfold from left to right, including interruptions, side-talk and brief acknowledgments.':s.insight,'step-count':`${String(index+1).padStart(2,'0')} / 08`}))document.getElementById(id).textContent=value;
   const axis=document.getElementById('insight-axis');axis.textContent=index<0?'μ':s.axis;axis.dataset.axis=s.axis;
   document.getElementById('state-eta').classList.toggle('positive',index>=3);document.getElementById('state-approval').classList.toggle('positive',index>=4);
   progress.setAttribute('aria-valuetext',index<0?'Ready to play':`Step ${index+1}: ${s.title}`);
  }
 }
 function tick(now){if(!playing)return;if(fraction>=1){if(!loopHoldUntil)loopHoldUntil=now+1000;if(now>=loopHoldUntil){fraction=0;index=-2;loopHoldUntil=0;loopCount++;demo.dataset.loopCount=String(loopCount);}}else{fraction=Math.min(1,fraction+(now-lastTime)/(durationSeconds*1000));}lastTime=now;render();pan();controls();frame=requestAnimationFrame(tick);}
 function resumeClock(){cancelAnimationFrame(frame);frame=null;if(playing&&!scrubbing){lastTime=performance.now();frame=requestAnimationFrame(tick);}}
 function start(){started=true;if(fraction>=1)fraction=0;playing=true;controls();resumeClock();}
 function pause(){loopHoldUntil=0;playing=false;cancelAnimationFrame(frame);frame=null;controls();}
 play.addEventListener('click',()=>{autoConsumed=true;playing?pause():start();});
 document.getElementById('demo-replay').addEventListener('click',()=>{autoConsumed=true;pause();fraction=0;render();pan();start();});
 function seek(value,follow=true){loopHoldUntil=0;autoConsumed=true;started=true;cancelAnimationFrame(frame);frame=null;fraction=Math.max(0,Math.min(1,value));render();if(follow)pan();controls();resumeClock();}
 function beginScrub(){autoConsumed=true;started=true;scrubbing=true;cancelAnimationFrame(frame);frame=null;controls();}
 function endScrub(){scrubbing=false;controls();resumeClock();}
 progress.addEventListener('pointerdown',beginScrub);
 progress.addEventListener('input',e=>seek(Number(e.target.value)/100));
 progress.addEventListener('change',endScrub);
 window.addEventListener('pointerup',()=>{if(scrubbing&&dragPointer===null)endScrub();});
 window.addEventListener('pointercancel',()=>{if(scrubbing&&dragPointer===null)endScrub();});
 function seekPointer(e){const bounds=axis.getBoundingClientRect();seek((e.clientX-bounds.left)/bounds.width,false);}
 axis.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragPointer=e.pointerId;beginScrub();axis.setPointerCapture(e.pointerId);axis.classList.add('dragging');seekPointer(e);e.preventDefault();});
 axis.addEventListener('pointermove',e=>{if(e.pointerId===dragPointer)seekPointer(e);});
 function endDrag(e){if(e.pointerId!==dragPointer)return;dragPointer=null;axis.classList.remove('dragging');if(axis.hasPointerCapture(e.pointerId))axis.releasePointerCapture(e.pointerId);endScrub();}
 axis.addEventListener('pointerup',endDrag);axis.addEventListener('pointercancel',endDrag);axis.addEventListener('lostpointercapture',()=>{if(dragPointer!==null){dragPointer=null;axis.classList.remove('dragging');endScrub();}});
 axis.addEventListener('keydown',e=>{let target;switch(e.key){case 'ArrowLeft':case 'ArrowDown':target=fraction-.025;break;case 'ArrowRight':case 'ArrowUp':target=fraction+.025;break;case 'PageDown':target=fraction-.1;break;case 'PageUp':target=fraction+.1;break;case 'Home':target=0;break;case 'End':target=1;break;default:return;}e.preventDefault();seek(target);});
 document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=null;}else if(playing){resumeClock();}});reduced.addEventListener('change',()=>{if(reduced.matches)pause();});
 if('ResizeObserver' in window)new ResizeObserver(()=>{render();if(playing)pan();}).observe(canvas);
 demo.querySelectorAll('.lane-avatar').forEach(avatar=>{avatar.insertAdjacentHTML('afterend','<span class="speaker-levels" aria-hidden="true">'+Array.from({length:5},(_,i)=>`<i style="--level:${i}"></i>`).join('')+'</span>');});
 render();controls();
 if('IntersectionObserver' in window){const auto=new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting&&!autoConsumed&&!reduced.matches){autoConsumed=true;start();}}},{threshold:.25});auto.observe(demo);
 const reveal=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){e.target.classList.add('visible');reveal.unobserve(e.target);}},{threshold:.08});document.querySelectorAll('.overview .section-heading,.track-grid,.framework .section-heading,.metric-explainer,#leaderboard .section-heading,.board,.start-grid').forEach(el=>{el.classList.add('reveal');reveal.observe(el);});}
})();

// Scroll-driven contour morph: the original mu shapes become the three portraits.
(()=>{
 const source=document.getElementById('hero-emblem');if(!source)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const overlay=document.createElement('div');overlay.className='identity-morph';overlay.setAttribute('aria-hidden','true');document.body.append(overlay);
 const colors={dad:[207,185,150],child:[208,158,166],agent:[145,168,232]},blue=[42,91,244],count=100;
 function contour(path,normalize){const length=path.getTotalLength(),b=path.getBBox(),size=Math.max(b.width,b.height);return Array.from({length:count},(_,i)=>{const p=path.getPointAtLength(length*i/count);return normalize?[(p.x-b.x-b.width/2)/size*40+20,(p.y-b.y-b.height/2)/size*40+20]:[p.x,p.y];});}
 function match(a,b){let best=null,cost=Infinity;for(const candidate of [b,b.slice().reverse()])for(let shift=0;shift<count;shift++){let distance=0;for(let i=0;i<count;i++){const q=candidate[(i+shift)%count];distance+=(a[i][0]-q[0])**2+(a[i][1]-q[1])**2;}if(distance<cost){cost=distance;best=Array.from({length:count},(_,i)=>candidate[(i+shift)%count]);}}return best;}
 const pieces=Array.from(source.querySelectorAll('path')).map(path=>{const person=path.dataset.identity,target=document.querySelector(`.lane-${person} .lane-avatar`),from=contour(path,true),to=match(from,contour(target.querySelector('.avatar-silhouette'),false)),bounds=path.getBBox();const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('identity-piece','morphing-portrait');svg.setAttribute('viewBox','0 0 40 40');const shape=document.createElementNS('http://www.w3.org/2000/svg','path');svg.append(shape);overlay.append(svg);return {person,target,from,to,bounds,svg,shape};});
 let queued=false;
 function draw(){queued=false;const origin=source.getBoundingClientRect(),last=pieces[2].target.getBoundingClientRect(),end=Math.max(400,last.top+scrollY-innerHeight*.72),p=Math.max(0,Math.min(1,scrollY/end));
  if(reduced.matches){source.style.opacity='1';overlay.style.visibility='hidden';pieces.forEach(o=>o.target.style.setProperty('--identity-appearance','1'));return;}
  const blend=p*p*(3-2*p),m=Math.max(0,Math.min(1,(p-.58)/.38)),morph=m*m*(3-2*m),arrival=Math.max(0,Math.min(1,(p-.96)/.04));source.style.opacity=p<.002?'1':'0';overlay.style.visibility=p>.002&&p<1?'visible':'hidden';
  pieces.forEach(({person,target,from,to,bounds,svg,shape},i)=>{const dest=target.getBoundingClientRect(),sx=origin.left+(bounds.x-118+bounds.width/2)/200*origin.width,sy=origin.top+(bounds.y-181+bounds.height/2)/178*origin.height,dx=dest.left+dest.width/2,dy=dest.top+dest.height/2,horizontal=1-Math.pow(1-p,6),edge=innerWidth<=640?15:dx,approach=Math.max(0,(p-.72)/.28),x=innerWidth<=640?sx+(edge-sx)*horizontal+(dx-edge)*approach:sx+(dx-sx)*horizontal+Math.sin(Math.PI*p)*(i-1)*12,y=sy+(dy-sy)*blend,startSize=Math.max(bounds.width,bounds.height)/200*origin.width,size=startSize+(36-startSize)*(1-Math.pow(1-p,3));
   const points=from.map((a,j)=>[a[0]+(to[j][0]-a[0])*morph,a[1]+(to[j][1]-a[1])*morph]);shape.setAttribute('d',points.map((v,j)=>(j?'L':'M')+v[0].toFixed(2)+' '+v[1].toFixed(2)).join(' ')+'Z');shape.style.fillOpacity=String(1-morph*.92);shape.style.strokeOpacity=String(morph);shape.style.strokeWidth='1.25';
   svg.style.width=size+'px';svg.style.height=size+'px';svg.style.transform=`translate(${x}px,${y}px) translate(-50%,-50%) rotate(${Math.sin(Math.PI*p)*(i-1)*10*(1-morph)}deg)`;svg.style.color=`rgb(${blue.map((c,k)=>Math.round(c+(colors[person][k]-c)*blend)).join(',')})`;svg.style.opacity=String(1-arrival);target.style.setProperty('--identity-appearance',String(arrival));
  });
 }
 function schedule(){if(!queued){queued=true;requestAnimationFrame(draw);}}
 addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);reduced.addEventListener('change',schedule);if('ResizeObserver'in window)new ResizeObserver(schedule).observe(document.getElementById('main'));draw();
})();
