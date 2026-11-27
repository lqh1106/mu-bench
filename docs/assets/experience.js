'use strict';

(() => {
  const demo = document.getElementById('demo');
  if (!demo) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const durationSeconds = 18;
  const progress = document.getElementById('demo-progress');
  const play = document.getElementById('demo-play');
  const canvas = document.getElementById('timeline-canvas');
  const viewport = document.getElementById('timeline-viewport');
  const axis = document.getElementById('time-axis');
  const stepPillsWrap = document.getElementById('step-pills');
  const cursorClock = document.getElementById('cursor-clock');
  const zoneBarge = document.getElementById('zone-barge');
  const zoneSidetalk = document.getElementById('zone-sidetalk');
  const zoneBackchannel = document.getElementById('zone-backchannel');

  const steps = [
    {
      at: 0.025,
      end: 0.17,
      who: 'dad',
      tag: 'To Agent · Primary Goal',
      text: 'Get us to the airport by 10.',
      axis: 'A',
      title: 'The shared task begins.',
      insight: 'Dad sets the destination and hard 10:00 deadline. Both stay bound to him throughout the session.',
      eta: 'Not checked',
      approval: 'Not requested',
      result: 'Main task registered (Dad)'
    },
    {
      at: 0.17,
      end: 0.32,
      who: 'agent',
      tag: 'Planning initial route',
      text: 'The fast route to…',
      tool: 'route_planner()',
      axis: 'T',
      title: 'The agent opens the main thread.',
      insight: 'Initial route planning begins — right before a second participant speaks over the reply.',
      eta: 'Not checked',
      approval: 'Not requested',
      result: 'route_planner() invoked'
    },
    {
      at: 0.24,
      end: 0.4,
      who: 'child',
      stacked: true,
      tag: 'Barge-in · To Agent',
      text: 'Can we stop by the park?',
      axis: 'A',
      title: 'A second voice barges in mid-reply.',
      insight: 'The child interrupts with a detour request that must be checked against Dad’s 10:00 constraint and authority.',
      eta: 'Checking…',
      approval: 'Awaiting Dad’s approval',
      result: 'Barge-in captured · main goal retained'
    },
    {
      at: 0.4,
      end: 0.61,
      who: 'agent',
      tag: 'Feasibility & permission check',
      text: 'We’d still reach the airport by 9:52. Should I add the stop?',
      tool: 'check_eta(via="park") → 09:52',
      axis: 'C',
      title: 'Verify feasibility before mutating state.',
      insight: 'ETA 09:52 satisfies the 10:00 deadline, so the agent asks the authorized decision-maker for approval.',
      eta: '09:52 (≤ 10:00)',
      approval: 'Awaiting Dad’s approval',
      result: 'check_eta() → 09:52 < 10:00'
    },
    {
      at: 0.61,
      end: 0.8,
      who: 'dad',
      tag: 'Approval · Constraint kept',
      text: 'Yes, as long as we arrive by 10.',
      axis: 'C',
      title: 'Conditional approval from the authority.',
      insight: 'Dad authorizes the park stop while explicitly keeping the 10:00 arrival requirement active.',
      eta: '09:52 (≤ 10:00)',
      approval: 'Approved by Dad',
      result: 'Authorized · deadline ≤ 10:00'
    },
    {
      at: 0.69,
      end: 0.8,
      who: 'child',
      stacked: true,
      tag: 'Peer side-talk · To Dad',
      text: 'Yay! Thanks, Dad!',
      axis: 'A',
      title: 'Peer side-talk is not an agent command.',
      insight: 'The child is addressing Dad, not issuing a new request. The agent must filter side-talk without derailing.',
      eta: '09:52 (≤ 10:00)',
      approval: 'Approved by Dad',
      result: 'Side-talk ignored · no new tool call'
    },
    {
      at: 0.8,
      end: 1.0,
      who: 'agent',
      tag: 'Main thread resumed',
      text: 'We’ll stop at the park first… then head to the airport.',
      tool: 'route_planner(via="park")',
      axis: 'T',
      title: 'Resume and commit the jointly valid plan.',
      insight: 'The agent executes the updated route including the park stop while preserving the airport destination.',
      eta: '09:52 (≤ 10:00)',
      approval: 'Approved by Dad',
      result: 'route_planner(via="park") committed'
    },
    {
      at: 0.89,
      end: 1.0,
      who: 'dad',
      stacked: true,
      tag: 'Backchannel',
      text: 'Mm-hmm.',
      axis: 'C',
      title: 'Brief backchannel leaves final state intact.',
      insight: 'A listener acknowledgment introduces no new obligation; all goals and constraints hold jointly.',
      eta: '09:52 (≤ 10:00)',
      approval: 'Approved by Dad',
      result: 'Joint Task Success ✓'
    }
  ];

  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[c]));

  steps.forEach((s, i) => {
    const wordsArr = s.text.split(' ');
    const el = document.createElement('article');
    el.className = `timeline-event event-${s.who}`;
    el.dataset.event = String(i);
    el.style.left = `${s.at * 100}%`;
    el.style.width = `${(s.end - s.at) * 100}%`;
    el.title = `Jump to ${(s.at * durationSeconds).toFixed(1)}s — ${s.title}`;
    el.innerHTML = `<span class="event-tag">${esc(s.tag)}</span>${s.tool ? `<code>${esc(s.tool)}</code>` : ''}<p class="spoken-text" aria-label="${esc(s.text)}"><span aria-hidden="true">“</span>${wordsArr
      .map((word, wi) => `<span class="spoken-word" aria-hidden="true">${esc(word)}${wi < wordsArr.length - 1 ? ' ' : ''}</span>`)
      .join('')}<span class="closing-quote" aria-hidden="true">”</span></p>${i === 1 ? '<span class="interrupted-tag">Interrupted at 4.3s</span>' : ''}`;
    document.getElementById(`lane-${s.who}`).append(el);

    const strip = document.createElement('span');
    strip.className = `audio-segment segment-${s.who}${s.stacked ? ' segment-stacked' : ''}`;
    strip.style.left = `${s.at * 100}%`;
    strip.style.width = `${(s.end - s.at) * 100}%`;
    document.getElementById('audio-strips').append(strip);

    if (stepPillsWrap) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'step-pill';
      btn.dataset.stepJump = String(i);
      btn.textContent = String(i + 1);
      btn.title = `Step ${i + 1}: ${s.title}`;
      btn.setAttribute('aria-label', `Jump to step ${i + 1}: ${s.title}`);
      stepPillsWrap.append(btn);
    }
  });

  const events = steps.map((step, eventId) => {
    const el = demo.querySelector(`[data-event="${eventId}"]`);
    return { step, el, words: Array.from(el.querySelectorAll('.spoken-word')), wordCount: -1 };
  });
  const stepButtons = stepPillsWrap ? Array.from(stepPillsWrap.querySelectorAll('.step-pill')) : [];

  let fraction = Math.max(0, Math.min(1, Number(new URLSearchParams(location.search).get('t')) || 0));
  let index = -2;
  let playing = false;
  let frame = null;
  let lastTime = 0;
  let autoConsumed = false;
  let started = false;
  let scrubbing = false;
  let dragPointer = null;
  let loopHoldUntil = 0;

  function controls() {
    demo.dataset.playing = String(playing);
    demo.dataset.scrubbing = String(scrubbing);
    document.getElementById('play-label').textContent = playing ? 'Pause' : fraction >= 1 ? 'Replay' : 'Play';
    document.getElementById('play-symbol').textContent = playing ? 'Ⅱ' : '▶';
    play.setAttribute('aria-label', playing ? 'Pause walkthrough' : fraction >= 1 ? 'Replay walkthrough' : 'Play walkthrough');
    document.getElementById('demo-status').textContent = scrubbing
      ? 'Scrubbing timeline…'
      : playing
      ? fraction >= 1
        ? 'Completed · looping'
        : 'Live trace playing'
      : fraction >= 1
      ? 'Joint outcome verified'
      : started
      ? 'Paused'
      : 'Ready to play';
  }

  function pan() {
    const x = axis.offsetLeft + axis.clientWidth * fraction;
    viewport.scrollLeft = Math.max(0, x - viewport.clientWidth * 0.7);
  }

  function render() {
    const next = steps.reduce((n, s, i) => (fraction >= s.at ? i : n), -1);
    demo.dataset.phase = String(next);
    const cursor = axis.offsetLeft + axis.clientWidth * fraction;
    const cursorEl = document.getElementById('time-cursor');
    cursorEl.style.left = `${cursor}px`;
    cursorEl.classList.toggle('cursor-end', fraction > 0.88);
    document.getElementById('time-elapsed').style.width = `${fraction * 100}%`;
    document.getElementById('audio-strips').style.clipPath = `inset(0 ${(1 - fraction) * 100}% 0 0)`;
    document.getElementById('retained-progress').style.width = `${fraction * 100}%`;
    document.getElementById('axis-handle').style.left = `${fraction * 100}%`;
    axis.setAttribute('aria-valuenow', String(Math.round(fraction * 1000) / 10));

    const exactSeconds = Math.min(durationSeconds, fraction * durationSeconds);
    const seconds = Math.floor(exactSeconds);
    if (cursorClock) cursorClock.textContent = `${exactSeconds.toFixed(1)}s`;
    axis.setAttribute('aria-valuetext', `00:${String(seconds).padStart(2, '0')} of 00:18`);
    document.getElementById('timeline-time').innerHTML = `00:${String(seconds).padStart(2, '0')} <span>/ 00:18</span>`;
    progress.value = String(fraction * 100);
    progress.style.setProperty('--progress', `${fraction * 100}%`);

    if (zoneBarge) zoneBarge.classList.toggle('zone-active', fraction >= 0.24 && fraction < 0.4);
    if (zoneSidetalk) zoneSidetalk.classList.toggle('zone-active', fraction >= 0.69 && fraction < 0.82);
    if (zoneBackchannel) zoneBackchannel.classList.toggle('zone-active', fraction >= 0.89);

    events.forEach((event) => {
      const { el, step, words } = event;
      const delivered = fraction >= step.at;
      const active = delivered && fraction < step.end;
      el.classList.toggle('delivered', delivered);
      el.classList.toggle('speaking', active);
      const spoken = Math.max(0, Math.min(1, (fraction - step.at) / (step.end - step.at)));
      const wordCount = delivered ? Math.min(words.length, Math.max(1, Math.ceil(spoken * words.length))) : 0;
      if (wordCount !== event.wordCount) {
        event.wordCount = wordCount;
        words.forEach((word, i) => {
          word.classList.toggle('revealed', i < wordCount);
          word.classList.toggle('word-current', i === wordCount - 1 && active);
        });
      }
      el.querySelector('.closing-quote').classList.toggle('revealed', spoken >= 1);
    });

    for (const person of ['dad', 'child', 'agent']) {
      const active = steps.some((step) => step.who === person && fraction >= step.at && fraction < step.end);
      demo.querySelector(`.lane-${person}`).classList.toggle('lane-speaking', active);
    }

    const intr = demo.querySelector('.interrupted-tag');
    if (intr) intr.classList.toggle('shown', fraction >= steps[2].at);
    document.getElementById('thread-status').textContent = fraction >= 0.8 ? 'Retained ✓' : 'Active';

    if (next !== index) {
      index = next;
      const activeIdx = Math.max(0, index);
      const s = steps[activeIdx];
      const updates = {
        'state-eta': s.eta,
        'state-approval': s.approval,
        'tool-result': index < 0 ? 'Click play or any card to inspect' : s.result,
        'insight-title': index < 0 ? 'Follow the shared full-duplex stream.' : s.title,
        'insight-text':
          index < 0
            ? 'Ghosted blocks preview upcoming turns; play or click any utterance to inspect real-time Attribution, Threading, and Coordination.'
            : s.insight,
        'step-count': `${String(activeIdx + 1).padStart(2, '0')} / 08`
      };
      for (const [id, value] of Object.entries(updates)) {
        const node = document.getElementById(id);
        if (node) node.textContent = value;
      }
      const axisEl = document.getElementById('insight-axis');
      axisEl.textContent = index < 0 ? 'μ' : s.axis;
      axisEl.dataset.axis = index < 0 ? 'A' : s.axis;
      document.getElementById('state-eta').classList.toggle('positive', index >= 3);
      document.getElementById('state-approval').classList.toggle('positive', index >= 4);
      stepButtons.forEach((b, bi) => b.classList.toggle('active', bi === index));
      progress.setAttribute('aria-valuetext', index < 0 ? 'Ready to play' : `Step ${index + 1}: ${s.title}`);
    }
  }

  function tick(now) {
    if (!playing) return;
    if (fraction >= 1) {
      if (!loopHoldUntil) loopHoldUntil = now + 1400;
      if (now >= loopHoldUntil) {
        fraction = 0;
        index = -2;
        loopHoldUntil = 0;
      }
    } else {
      fraction = Math.min(1, fraction + (now - lastTime) / (durationSeconds * 1000));
    }
    lastTime = now;
    render();
    pan();
    controls();
    frame = requestAnimationFrame(tick);
  }

  function resumeClock() {
    cancelAnimationFrame(frame);
    frame = null;
    if (playing && !scrubbing) {
      lastTime = performance.now();
      frame = requestAnimationFrame(tick);
    }
  }

  function start() {
    started = true;
    if (fraction >= 1) fraction = 0;
    playing = true;
    controls();
    resumeClock();
  }

  function pause() {
    loopHoldUntil = 0;
    playing = false;
    cancelAnimationFrame(frame);
    frame = null;
    controls();
  }

  function seek(value, follow = true) {
    loopHoldUntil = 0;
    autoConsumed = true;
    started = true;
    cancelAnimationFrame(frame);
    frame = null;
    fraction = Math.max(0, Math.min(1, value));
    render();
    if (follow) pan();
    controls();
    resumeClock();
  }

  play.addEventListener('click', () => {
    autoConsumed = true;
    playing ? pause() : start();
  });

  document.getElementById('demo-replay').addEventListener('click', () => {
    autoConsumed = true;
    pause();
    fraction = 0;
    render();
    pan();
    start();
  });

  const emblemWrap = document.querySelector('.hero-emblem-wrap');
  if (emblemWrap) {
    emblemWrap.addEventListener('click', () => {
      document.getElementById('experience').scrollIntoView({ behavior: reduced.matches ? 'auto' : 'smooth' });
    });
  }

  events.forEach(({ step, el }) => {
    el.addEventListener('click', () => {
      seek(Math.min(0.995, step.at + 0.03), true);
    });
  });

  stepButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const s = steps[Number(btn.dataset.stepJump)];
      if (s) seek(Math.min(0.995, s.at + 0.03), true);
    });
  });

  function beginScrub() {
    autoConsumed = true;
    started = true;
    scrubbing = true;
    cancelAnimationFrame(frame);
    frame = null;
    controls();
  }

  function endScrub() {
    scrubbing = false;
    controls();
    resumeClock();
  }

  progress.addEventListener('pointerdown', beginScrub);
  progress.addEventListener('input', (e) => seek(Number(e.target.value) / 100));
  progress.addEventListener('change', endScrub);
  window.addEventListener('pointerup', () => {
    if (scrubbing && dragPointer === null) endScrub();
  });
  window.addEventListener('pointercancel', () => {
    if (scrubbing && dragPointer === null) endScrub();
  });

  function seekPointer(e) {
    const bounds = axis.getBoundingClientRect();
    seek((e.clientX - bounds.left) / bounds.width, false);
  }

  axis.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    dragPointer = e.pointerId;
    beginScrub();
    axis.setPointerCapture(e.pointerId);
    seekPointer(e);
    e.preventDefault();
  });

  axis.addEventListener('pointermove', (e) => {
    if (e.pointerId === dragPointer) seekPointer(e);
  });

  function endDrag(e) {
    if (e.pointerId !== dragPointer) return;
    dragPointer = null;
    if (axis.hasPointerCapture(e.pointerId)) axis.releasePointerCapture(e.pointerId);
    endScrub();
  }

  axis.addEventListener('pointerup', endDrag);
  axis.addEventListener('pointercancel', endDrag);
  axis.addEventListener('lostpointercapture', () => {
    if (dragPointer !== null) {
      dragPointer = null;
      endScrub();
    }
  });

  axis.addEventListener('keydown', (e) => {
    let target;
    switch (e.key) {
      case 'ArrowLeft':
      case 'ArrowDown':
        target = fraction - 0.025;
        break;
      case 'ArrowRight':
      case 'ArrowUp':
        target = fraction + 0.025;
        break;
      case 'PageDown':
        target = fraction - 0.1;
        break;
      case 'PageUp':
        target = fraction + 0.1;
        break;
      case 'Home':
        target = 0;
        break;
      case 'End':
        target = 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    seek(target);
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(frame);
      frame = null;
    } else if (playing) {
      resumeClock();
    }
  });

  reduced.addEventListener('change', () => {
    if (reduced.matches) pause();
  });

  if ('ResizeObserver' in window) {
    new ResizeObserver(() => {
      render();
      if (playing) pan();
    }).observe(canvas);
  }

  demo.querySelectorAll('.lane-avatar').forEach((avatar) => {
    avatar.insertAdjacentHTML(
      'afterend',
      '<span class="speaker-levels" aria-hidden="true">' + Array.from({ length: 5 }, (_, i) => `<i style="--level:${i}"></i>`).join('') + '</span>'
    );
  });

  render();
  controls();

  if ('IntersectionObserver' in window) {
    const auto = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting && !autoConsumed && !reduced.matches) {
            autoConsumed = true;
            start();
          }
        }
      },
      { threshold: 0.25 }
    );
    auto.observe(demo);
  }
})();

// Scroll-driven contour morph: #hero-emblem detaches and docks into the three speaker portraits without hiding resting avatars at scrollY=0.
(() => {
  const source = document.getElementById('hero-emblem');
  if (!source) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const overlay = document.createElement('div');
  overlay.className = 'identity-morph';
  overlay.setAttribute('aria-hidden', 'true');
  document.body.append(overlay);

  const colors = { dad: [168, 107, 20], child: [190, 67, 90], agent: [37, 87, 240] };
  const blue = [37, 87, 240];
  const count = 96;

  function contour(path, normalize) {
    const length = path.getTotalLength();
    const b = path.getBBox();
    const size = Math.max(b.width, b.height);
    return Array.from({ length: count }, (_, i) => {
      const p = path.getPointAtLength((length * i) / count);
      return normalize ? [((p.x - b.x - b.width / 2) / size) * 40 + 20, ((p.y - b.y - b.height / 2) / size) * 40 + 20] : [p.x, p.y];
    });
  }

  function match(a, b) {
    let best = null;
    let cost = Infinity;
    for (const candidate of [b, b.slice().reverse()]) {
      for (let shift = 0; shift < count; shift++) {
        let distance = 0;
        for (let i = 0; i < count; i++) {
          const q = candidate[(i + shift) % count];
          distance += (a[i][0] - q[0]) ** 2 + (a[i][1] - q[1]) ** 2;
        }
        if (distance < cost) {
          cost = distance;
          best = Array.from({ length: count }, (_, i) => candidate[(i + shift) % count]);
        }
      }
    }
    return best;
  }

  const pieces = Array.from(source.querySelectorAll('path')).map((path) => {
    const person = path.dataset.identity;
    const target = document.querySelector(`.lane-${person} .lane-avatar`);
    const from = contour(path, true);
    const to = match(from, contour(target.querySelector('.avatar-silhouette'), false));
    const bounds = path.getBBox();
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('identity-piece', 'morphing-portrait');
    svg.setAttribute('viewBox', '0 0 40 40');
    const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    svg.append(shape);
    overlay.append(svg);
    return { person, target, from, to, bounds, svg, shape };
  });

  let queued = false;

  function draw() {
    queued = false;
    if (reduced.matches) {
      source.style.opacity = '1';
      overlay.style.visibility = 'hidden';
      pieces.forEach((o) => o.target.style.setProperty('--identity-appearance', '1'));
      return;
    }

    const origin = source.getBoundingClientRect();
    const expSection = document.getElementById('experience');
    const headerH = document.querySelector('.header')?.getBoundingClientRect().height || 69;
    const expTopDoc = expSection ? expSection.getBoundingClientRect().top + scrollY : 600;
    const end = Math.max(120, expTopDoc - headerH - 12);
    const p = Math.max(0, Math.min(1, scrollY / end));

    if (p <= 0.01) {
      source.style.opacity = '1';
      overlay.style.visibility = 'hidden';
      pieces.forEach((o) => o.target.style.setProperty('--identity-appearance', '0.94'));
      return;
    }

    if (p >= 0.995) {
      source.style.opacity = '0';
      overlay.style.visibility = 'hidden';
      pieces.forEach((o) => o.target.style.setProperty('--identity-appearance', '1'));
      return;
    }

    const blend = p * p * (3 - 2 * p);
    const m = Math.max(0, Math.min(1, (p - 0.52) / 0.42));
    const morph = m * m * (3 - 2 * m);
    const arrival = Math.max(0.22, Math.min(1, (p - 0.85) / 0.15));

    source.style.opacity = '0';
    overlay.style.visibility = 'visible';

    pieces.forEach(({ person, target, from, to, bounds, svg, shape }, i) => {
      const dest = target.getBoundingClientRect();
      const sx = origin.left + ((bounds.x - 118 + bounds.width / 2) / 200) * origin.width;
      const sy = origin.top + ((bounds.y - 181 + bounds.height / 2) / 178) * origin.height;
      const dx = dest.left + dest.width / 2;
      const dy = dest.top + dest.height / 2;
      const x = sx + (dx - sx) * blend + Math.sin(Math.PI * p) * (i - 1) * 12;
      const y = sy + (dy - sy) * blend;
      const startSize = (Math.max(bounds.width, bounds.height) / 200) * origin.width;
      const size = startSize + (36 - startSize) * (1 - Math.pow(1 - p, 3));

      const points = from.map((a, j) => [a[0] + (to[j][0] - a[0]) * morph, a[1] + (to[j][1] - a[1]) * morph]);
      shape.setAttribute('d', points.map((v, j) => (j ? 'L' : 'M') + v[0].toFixed(2) + ' ' + v[1].toFixed(2)).join(' ') + 'Z');
      shape.style.fillOpacity = String(1 - morph * 0.88);
      shape.style.strokeOpacity = String(morph);
      shape.style.strokeWidth = '1.3';

      svg.style.width = size + 'px';
      svg.style.height = size + 'px';
      svg.style.transform = `translate(${x}px,${y}px) translate(-50%,-50%) rotate(${Math.sin(Math.PI * p) * (i - 1) * 9 * (1 - morph)}deg)`;
      svg.style.color = `rgb(${blue.map((c, k) => Math.round(c + (colors[person][k] - c) * blend)).join(',')})`;
      const dockFade = Math.max(0, Math.min(1, (p - 0.9) / 0.1));
      svg.style.opacity = String(1 - dockFade);
      target.style.setProperty('--identity-appearance', String(arrival));
    });
  }

  function schedule() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(draw);
    }
  }

  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule);
  document.getElementById('timeline-viewport')?.addEventListener('scroll', schedule, { passive: true });
  reduced.addEventListener('change', schedule);
  if ('ResizeObserver' in window) new ResizeObserver(schedule).observe(document.getElementById('main'));
  draw();
})();
