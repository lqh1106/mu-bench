'use strict';

// Choreographed Scroll Reveal, Spotlight Cards & Hero Scale Count-Up
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  if (!reduced.matches) {
    document.documentElement.classList.add('js-motion');
  }

  const revealTargets = Array.from(document.querySelectorAll('[data-reveal]'));
  if (revealTargets.length) {
    const show = (el) => el.classList.add('is-visible');
    revealTargets.forEach((el) => {
      if (el.getBoundingClientRect().top < innerHeight * 1.12) show(el);
    });
    if ('IntersectionObserver' in window && !reduced.matches) {
      const obs = new IntersectionObserver(
        (entries) => {
          for (const e of entries) {
            if (e.isIntersecting) {
              show(e.target);
              obs.unobserve(e.target);
            }
          }
        },
        { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
      );
      revealTargets.forEach((el) => {
        if (!el.classList.contains('is-visible')) obs.observe(el);
      });
    } else {
      revealTargets.forEach(show);
    }
  }

  // Interactive Pointer Spotlight on Feature Cards
  if (!reduced.matches) {
    document.querySelectorAll('.spotlight-card').forEach((card) => {
      card.addEventListener('pointermove', (e) => {
        const r = card.getBoundingClientRect();
        card.style.setProperty('--mx', `${e.clientX - r.left}px`);
        card.style.setProperty('--my', `${e.clientY - r.top}px`);
      });
    });
  }

  // Subtle Count-Up on Hero Scale Numbers
  if (!reduced.matches) {
    document.querySelectorAll('[data-count]').forEach((el) => {
      const target = Number(el.dataset.count);
      if (!target) return;
      const start = performance.now();
      const dur = 820;
      const step = (now) => {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = String(Math.round(target * eased));
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }
})();

// Leaderboard 2.0 (Chart + Full Table 2 Matrix + Pure JTS Overlay)
(() => {
  const dataset = JSON.parse(document.getElementById('leaderboard-data').textContent);
  const chart = document.getElementById('rank-chart');
  const rows = document.getElementById('chart-rows');
  const matrixBody = document.getElementById('matrix-tbody');
  const chartPane = document.getElementById('chart-view-pane');
  const tablePane = document.getElementById('table-view-pane');
  const pureBtn = document.getElementById('pure-jts-toggle');
  const pureLegend = document.getElementById('pure-legend-item');
  const metricSelect = document.getElementById('chart-metric');

  const state = {
    view: 'chart',
    track: 'tool_free',
    metric: 0,
    pureMode: false,
    family: 'all',
    query: '',
    tableSort: 'tf_0',
    tableAsc: false
  };

  const metricNames = ['Joint Task Success', 'Attribution', 'Threading', 'Coordination'];
  const shortMetrics = ['JTS', 'A', 'T', 'C'];
  let visible = false;

  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[c]));

  function getChartScore(m) {
    if (state.pureMode && state.track === 'tool_augmented' && state.metric === 0) {
      return typeof m.pure_jts === 'number' ? m.pure_jts : null;
    }
    return m[state.track] ? m[state.track][state.metric] : null;
  }

  function filteredChartModels() {
    return dataset.models
      .filter((m) => {
        const val = getChartScore(m);
        if (val === null || val === undefined) return false;
        if (state.family !== 'all' && m.family !== state.family) return false;
        if (state.query && !m.name.toLowerCase().includes(state.query)) return false;
        return true;
      })
      .sort((a, b) => getChartScore(b) - getChartScore(a));
  }

  function renderChart() {
    chart.classList.remove('is-expanded');
    const models = filteredChartModels();
    const isPure = state.pureMode && state.track === 'tool_augmented' && state.metric === 0;
    pureLegend.hidden = !isPure;

    if (!models.length) {
      rows.innerHTML = '<p class="chart-empty">No evaluated systems match your current filter.</p>';
    } else {
      rows.innerHTML = models
        .map((m, i) => {
          const score = getChartScore(m);
          const rawArr = m[state.track];
          const stdJts = rawArr ? rawArr[0] : score;
          const delta = isPure ? (score - stdJts).toFixed(1) : null;
          const labelPos = isPure ? Math.max(score, stdJts) : score;
          const famLabel = m.family === 'proprietary' ? 'Proprietary' : 'Open';
          const delay = Math.min(i * 26, 260);

          const atcHtml = rawArr
            ? `<div class="row-atc" aria-label="ATC diagnostics">
                <span class="atc-pill ${state.metric === 1 ? 'active-metric' : ''}">A <b>${rawArr[1].toFixed(1)}</b></span>
                <span class="atc-pill ${state.metric === 2 ? 'active-metric' : ''}">T <b>${rawArr[2].toFixed(1)}</b></span>
                <span class="atc-pill ${state.metric === 3 ? 'active-metric' : ''}">C <b>${rawArr[3].toFixed(1)}</b></span>
              </div>`
            : '<div class="row-atc"></div>';

          return `<div class="chart-row" role="listitem" aria-label="Rank ${i + 1}, ${esc(m.name)}: ${score.toFixed(2)} percent">
            <div class="chart-model-wrap">
              <span class="rank-num">${String(i + 1).padStart(2, '0')}</span>
              <span class="chart-model" title="${esc(m.name)}">${esc(m.name)}</span>
              <span class="fam-tag ${m.family}">${famLabel}</span>
            </div>
            <div class="bar-slot" style="--score:${score}%;--ghost-score:${stdJts}%;--label-pos:${labelPos}%;--delay:${delay}ms">
              ${isPure ? `<span class="bar-ghost" title="Standard JTS: ${stdJts.toFixed(2)}%"></span>` : ''}
              <span class="bar-fill ${m.family} ${isPure ? 'pure-mode' : ''}" title="${shortMetrics[state.metric]}: ${score.toFixed(2)}%"></span>
              <span class="bar-value">${score.toFixed(2)}${isPure ? `<em class="bar-delta">${delta} pp</em>` : ''}</span>
            </div>
            ${atcHtml}
          </div>`;
        })
        .join('');
    }

    document.getElementById('result-count').textContent = `${models.length} evaluated ${models.length === 1 ? 'system' : 'systems'} · scores in %`;
    document.getElementById('chart-caption').textContent = isPure
      ? 'Pure JTS (n = 0 unauthorized calls) vs. Standard JTS (%)'
      : `${metricNames[state.metric]} (%) · ${state.track === 'tool_free' ? 'Tool-Free (392 cases)' : 'Tool-Augmented (408 cases)'}`;

    if (visible) {
      requestAnimationFrame(() => requestAnimationFrame(() => chart.classList.add('is-expanded')));
    }
  }

  function getTableSortVal(m, key) {
    if (key === 'pure') return typeof m.pure_jts === 'number' ? m.pure_jts : null;
    const [prefix, idxStr] = key.split('_');
    const idx = Number(idxStr);
    const arr = prefix === 'tf' ? m.tool_free : m.tool_augmented;
    return arr ? arr[idx] : null;
  }

  function renderTable() {
    const models = dataset.models.filter((m) => {
      if (state.family !== 'all' && m.family !== state.family) return false;
      if (state.query && !m.name.toLowerCase().includes(state.query)) return false;
      return true;
    });

    if (!models.length) {
      matrixBody.innerHTML = '<tr><td colspan="10" class="chart-empty">No evaluated systems match your current filter.</td></tr>';
      document.getElementById('result-count').textContent = '0 evaluated systems';
      return;
    }

    const best = {
      prop_tf0: Math.max(...dataset.models.filter((m) => m.family === 'proprietary' && m.tool_free).map((m) => m.tool_free[0])),
      open_tf0: Math.max(...dataset.models.filter((m) => m.family === 'open' && m.tool_free).map((m) => m.tool_free[0])),
      ta0: Math.max(...dataset.models.filter((m) => m.tool_augmented).map((m) => m.tool_augmented[0])),
      pure: Math.max(...dataset.models.filter((m) => typeof m.pure_jts === 'number').map((m) => m.pure_jts))
    };

    const sorter = (a, b) => {
      const va = getTableSortVal(a, state.tableSort);
      const vb = getTableSortVal(b, state.tableSort);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      return state.tableAsc ? va - vb : vb - va;
    };

    const groups = state.family === 'all'
      ? [
          { title: 'Proprietary Voice Agents', items: models.filter((m) => m.family === 'proprietary').sort(sorter) },
          { title: 'Open-Weight Speech Models', items: models.filter((m) => m.family === 'open').sort(sorter) }
        ]
      : [{ title: state.family === 'proprietary' ? 'Proprietary Voice Agents' : 'Open-Weight Speech Models', items: models.sort(sorter) }];

    const fmtCell = (val, cls = '', isBest = false) => {
      if (val === null || val === undefined) return `<td class="${cls} missing">—</td>`;
      return `<td class="${cls}${isBest ? ' best' : ''}">${val.toFixed(2)}</td>`;
    };

    matrixBody.innerHTML = groups
      .filter((g) => g.items.length)
      .map((g) => {
        const headerRow = `<tr class="family-row"><th colspan="10">${esc(g.title)}</th></tr>`;
        const dataRows = g.items
          .map((m) => {
            const tf = m.tool_free || [null, null, null, null];
            const ta = m.tool_augmented || [null, null, null, null];
            const tfBest = m.family === 'proprietary' ? tf[0] === best.prop_tf0 : tf[0] === best.open_tf0;
            return `<tr class="data-row">
              <th scope="row" class="model-cell">${esc(m.name)}</th>
              ${fmtCell(tf[0], 'jts-cell', tfBest)}
              ${fmtCell(tf[1])}
              ${fmtCell(tf[2])}
              ${fmtCell(tf[3])}
              ${fmtCell(ta[0], 'jts-cell track-separator', ta[0] === best.ta0)}
              ${fmtCell(m.pure_jts ?? null, 'pure-cell', m.pure_jts === best.pure)}
              ${fmtCell(ta[1])}
              ${fmtCell(ta[2])}
              ${fmtCell(ta[3])}
            </tr>`;
          })
          .join('');
        return headerRow + dataRows;
      })
      .join('');

    document.getElementById('result-count').textContent = `${models.length} systems in Paper Table 2 matrix · "—" indicates unevaluated track`;
  }

  function syncControls() {
    document.querySelectorAll('[data-chart-track]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.chartTrack === state.track));
    });
    document.querySelectorAll('[data-board-view]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.boardView === state.view));
    });
    const pureActive = state.pureMode && state.track === 'tool_augmented' && state.metric === 0;
    pureBtn.setAttribute('aria-pressed', String(pureActive));
    metricSelect.value = String(state.metric);
    chartPane.hidden = state.view !== 'chart';
    tablePane.hidden = state.view !== 'table';
    document.getElementById('metric-selector-wrap').style.display = state.view === 'chart' ? '' : 'none';
  }

  function renderAll() {
    syncControls();
    if (state.view === 'chart') renderChart();
    else renderTable();
  }

  document.querySelectorAll('[data-chart-track]').forEach((b) => {
    b.addEventListener('click', () => {
      state.track = b.dataset.chartTrack;
      if (state.track === 'tool_free') state.pureMode = false;
      state.view = 'chart';
      renderAll();
    });
  });

  document.querySelectorAll('[data-board-view]').forEach((b) => {
    b.addEventListener('click', () => {
      state.view = b.dataset.boardView;
      renderAll();
    });
  });

  pureBtn.addEventListener('click', () => {
    const next = !(state.pureMode && state.track === 'tool_augmented' && state.metric === 0 && state.view === 'chart');
    state.pureMode = next;
    if (next) {
      state.view = 'chart';
      state.track = 'tool_augmented';
      state.metric = 0;
    }
    renderAll();
  });

  metricSelect.addEventListener('change', (e) => {
    state.metric = Number(e.target.value);
    if (state.metric !== 0) state.pureMode = false;
    renderAll();
  });

  document.getElementById('chart-family').addEventListener('change', (e) => {
    state.family = e.target.value;
    renderAll();
  });

  document.getElementById('chart-search').addEventListener('input', (e) => {
    state.query = e.target.value.trim().toLowerCase();
    renderAll();
  });

  document.querySelectorAll('.metric-sort').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.sort;
      if (state.tableSort === key) {
        state.tableAsc = !state.tableAsc;
      } else {
        state.tableSort = key;
        state.tableAsc = false;
      }
      document.querySelectorAll('.metric-sort').forEach((b) => {
        const active = b === btn;
        b.classList.toggle('active', active);
        b.querySelector('.sort-mark').textContent = active ? (state.tableAsc ? '↑' : '↓') : '↕';
      });
      renderTable();
    });
  });

  document.getElementById('download').addEventListener('click', () => {
    const columns = ['System', 'Family', 'TF_JTS', 'TF_A', 'TF_T', 'TF_C', 'TA_JTS', 'TA_Pure_JTS', 'TA_A', 'TA_T', 'TA_C', 'TA_Unauth_Episode_Pct'];
    const data = [
      columns,
      ...dataset.models.map((m) => [
        m.name,
        m.family,
        ...(m.tool_free ?? ['', '', '', '']),
        ...(m.tool_augmented ? [m.tool_augmented[0], m.pure_jts ?? '', m.tool_augmented[1], m.tool_augmented[2], m.tool_augmented[3], m.unauth_episode_pct ?? ''] : ['', '', '', '', '', ''])
      ])
    ];
    const csv = data.map((r) => r.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(',')).join('\r\n');
    const a = document.createElement('a');
    const url = URL.createObjectURL(new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' }));
    a.href = url;
    a.download = 'mu-bench-scores.csv';
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  renderAll();

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            visible = true;
            chart.classList.add('is-expanded');
          }
        }
      },
      { threshold: 0.08 }
    ).observe(chart);
  } else {
    visible = true;
    chart.classList.add('is-expanded');
  }
})();

// Cumulative Capability Ablation Interactive Stepper
(() => {
  const stages = [
    {
      title: 'Stage 0 · Full-Duplex Baseline (19.39% JTS)',
      body: 'Overlapping speech and barge-ins severely depress Attribution (36.89%) and Threading (35.95%), even though passive Coordination reaches 87.17%.'
    },
    {
      title: 'Stage 1 · + Serialized Turns (26.79% JTS · +7.40 pp)',
      body: 'Converting overlapping speech into sequential turns delivers over 60% of the total ablation gain, lifting Attribution (+16.72 pp) and Threading (+15.08 pp) in tandem.'
    },
    {
      title: 'Stage 2 · + Identity Metadata (29.59% JTS · +2.80 pp)',
      body: 'Supplying turn-level speaker, addressee, and request-owner tags raises downstream Attribution to 63.35% and Threading to 53.53%, yet most multi-user sessions still fail.'
    },
    {
      title: 'Stage 3 · + Structured Semantics (31.38% JTS · +1.79 pp)',
      body: 'Even with serialized turns, explicit speaker routing, and structured goal/constraint frames, 68.62% of sessions remain unsolved — isolating multi-party planning as an open challenge.'
    }
  ];

  const callout = document.getElementById('ablation-callout');
  const btns = document.querySelectorAll('.ablation-btn');
  const barGroups = document.querySelectorAll('.ablation-bars g[data-bar]');
  if (!callout || !btns.length) return;

  function setStage(idx) {
    btns.forEach((b, i) => {
      b.classList.toggle('active', i === idx);
      b.setAttribute('aria-selected', String(i === idx));
    });
    barGroups.forEach((g, i) => g.classList.toggle('active', i === idx));
    callout.querySelector('strong').textContent = stages[idx].title;
    callout.querySelector('p').textContent = stages[idx].body;
  }

  btns.forEach((b) => b.addEventListener('click', () => setStage(Number(b.dataset.stage))));
  barGroups.forEach((g) => g.addEventListener('click', () => setStage(Number(g.dataset.bar))));
})();

// Interactive 20-Domain Explorer (51 TA + 49 TF = 100 base scenarios)
(() => {
  const domains = [
    { name: 'Aviation & Airport', cluster: 'mobility', ta: 3, tf: 2, summary: 'Separate rebooking, seat & cancellation authority, gate broadcast filtering.' },
    { name: 'Travel Documents', cluster: 'mobility', ta: 3, tf: 2, summary: 'Passport correction, multi-user renewal, parental authority & side-talk privacy.' },
    { name: 'In-Vehicle Collaboration', cluster: 'mobility', ta: 3, tf: 2, summary: 'Detour & destination updates, music side-talk, latecomer summary & speaking equity.' },
    { name: 'Public Services', cluster: 'mobility', ta: 2, tf: 3, summary: 'Accessible transit & interpreter booking, borrowing privacy, resident turn equity.' },
    { name: 'Retail & E-Commerce', cluster: 'commerce', ta: 4, tf: 1, summary: 'Cancellation withdrawal, address & quantity edits, return side-talk filtering.' },
    { name: 'Telecommunications', cluster: 'commerce', ta: 5, tf: 0, summary: 'Data top-up, lost-number fix, reboot objection, roaming relay & outage waiver.' },
    { name: 'Finance & Billing', cluster: 'commerce', ta: 3, tf: 2, summary: 'Auto-debit conflict, currency split, invoice fix, benefit privacy & refusal resumption.' },
    { name: 'Dining & Hospitality', cluster: 'commerce', ta: 3, tf: 2, summary: 'Party size revision, ordering side-talk, bill split, allergy & roommate privacy.' },
    { name: 'Housing & Relocation', cluster: 'commerce', ta: 4, tf: 1, summary: 'Filter coordination, commute comparison, viewing slots, guarantor vs. tenant consent.' },
    { name: 'Shared Smart Home', cluster: 'home', ta: 3, tf: 2, summary: 'Multi-owner timers, media control, shared shopping list, thermostat & visitor access.' },
    { name: 'Household Coordination', cluster: 'home', ta: 2, tf: 3, summary: 'Pickup division, chore assignment, eldercare scheduling & calendar privacy.' },
    { name: 'Healthcare & Caregiving', cluster: 'home', ta: 2, tf: 3, summary: 'Appointment coordination, medication source logging, visitor privacy & guidelines.' },
    { name: 'Events & Entertainment', cluster: 'home', ta: 4, tf: 1, summary: 'Ticket allocation, collaborative playlist, accessible seating & parental limits.' },
    { name: 'Casual Conversation', cluster: 'home', ta: 0, tf: 5, summary: 'Story continuation, backchannels, roommate side-talk, digression return & inclusion.' },
    { name: 'Emotional Support', cluster: 'home', ta: 0, tf: 5, summary: 'Dual-party needs, comfort target attribution, emotion source & grounded empathy.' },
    { name: 'Meeting Collaboration', cluster: 'work', ta: 4, tf: 1, summary: 'Vote recording, owner/deadline edits, dissent logging, quoted vs. personal vote.' },
    { name: 'IT Support', cluster: 'work', ta: 2, tf: 3, summary: 'Service restart authority, rollback target fix, credential privacy & multi-thread triage.' },
    { name: 'Education & Learning', cluster: 'work', ta: 2, tf: 3, summary: 'Student answer grading, project division, peer whisper filtering & lecture backchannels.' },
    { name: 'Emergency & Incident', cluster: 'work', ta: 2, tf: 3, summary: 'Missing person & roll-call logging, evacuation clarification & shelter negotiation.' },
    { name: 'Knowledge QA', cluster: 'work', ta: 0, tf: 5, summary: 'Answer attribution, coreference, expert counterexample, translation & child-level QA.' }
  ];

  const grid = document.getElementById('domain-grid');
  const chips = document.querySelectorAll('.domain-chip');
  if (!grid) return;

  function renderDomains(cluster) {
    const items = cluster === 'all' ? domains : domains.filter((d) => d.cluster === cluster);
    grid.innerHTML = items
      .map(
        (d, idx) => `<article class="domain-item" role="listitem" style="--d-delay:${Math.min(idx * 22, 240)}ms">
          <div class="domain-item-top">
            <strong>${d.name}</strong>
            <span class="domain-split">${d.ta} TA · ${d.tf} TF</span>
          </div>
          <p>${d.summary}</p>
        </article>`
      )
      .join('');
  }

  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      chips.forEach((c) => c.classList.toggle('active', c === chip));
      renderDomains(chip.dataset.cluster);
    });
  });

  renderDomains('all');
})();

// Tabbed Quick-Start Terminal & Copy Buttons
(() => {
  const snippets = {
    install: {
      code: `export MUBENCH_REPO="$HOME/mu-bench"
git clone https://github.com/lqh1106/mu-bench.git "$MUBENCH_REPO"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets`,
      caption: 'Linux · Python 3.10–3.12 · Validates all 800 cases, deterministic voice bindings, and runtime contracts.'
    },
    native: {
      code: `export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO" && set -a && source .env && set +a
.venv/bin/python -m mubench run --config local.native.json \\
  --subset tool_free \\
  --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1`,
      caption: 'Runs one Tool-Free native GPU audio smoke case after deploying CosyVoice3 TTS, Whisper ASR, and model service.'
    },
    cloud: {
      code: `export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO" && set -a && source .env && set +a
.venv/bin/python -m mubench run --config local.closed.json \\
  --subset tool_augmented \\
  --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1`,
      caption: 'Runs one Tool-Augmented realtime cloud audio smoke case and writes sealed scores + stereo audio under formal/.'
    },
    text: {
      code: `export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp configs/run.text.example.json local.text.json
.venv/bin/python -m mubench run --config local.text.json \\
  --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1`,
      caption: 'GPU-free streaming text diagnostic track using speaker-labeled utterances, prefix interruption, and identical tools.'
    }
  };

  const codeEl = document.getElementById('quickstart');
  const captionEl = document.getElementById('code-caption');
  const statusEl = document.getElementById('copy-status');

  document.querySelectorAll('.code-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const s = snippets[tab.dataset.snippet];
      if (!s) return;
      document.querySelectorAll('.code-tab').forEach((t) => {
        t.classList.toggle('active', t === tab);
        t.setAttribute('aria-selected', String(t === tab));
      });
      codeEl.textContent = s.code;
      captionEl.textContent = s.caption;
    });
  });

  function bindCopy(btnId, targetId, label) {
    const btn = document.getElementById(btnId);
    const target = document.getElementById(targetId);
    if (!btn || !target) return;
    const orig = btn.textContent;
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(target.textContent);
        btn.textContent = 'Copied ✓';
        if (statusEl) statusEl.textContent = `${label} copied to clipboard.`;
        setTimeout(() => {
          btn.textContent = orig;
        }, 1800);
      } catch {
        if (statusEl) statusEl.textContent = 'Select the code block to copy manually.';
      }
    });
  }

  bindCopy('copy', 'quickstart', 'Quick-start commands');
  bindCopy('copy-bib', 'bibtex-code', 'BibTeX citation');
})();

// Exact Sticky Header Height Sync (--header-h) & Deterministic ScrollSpy
(() => {
  const header = document.querySelector('.header');
  const navLinks = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  const sections = navLinks.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);

  function syncHeaderHeight() {
    if (!header) return 68;
    const h = Math.floor(header.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--header-h', `${h}px`);
    return h;
  }

  syncHeaderHeight();
  if (header && 'ResizeObserver' in window) {
    new ResizeObserver(syncHeaderHeight).observe(header);
  }
  addEventListener('resize', syncHeaderHeight);

  // Precise smooth anchor navigation so every section lands flush beneath the sticky header
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (!href) return;
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (href === '#') {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' });
        history.replaceState(null, '', location.pathname + location.search);
        return;
      }
      const target = document.querySelector(href);
      if (!target) return;
      e.preventDefault();
      const headerH = syncHeaderHeight();
      const top = Math.max(0, Math.round(target.getBoundingClientRect().top + window.scrollY - headerH));
      window.scrollTo({ top, behavior: reduced ? 'auto' : 'smooth' });
      history.replaceState(null, '', href);
    });
  });

  if (!sections.length) return;

  let spyTicking = false;
  function updateActiveNav() {
    spyTicking = false;
    const headerH = header ? header.getBoundingClientRect().height : 69;
    if (scrollY < 120) {
      navLinks.forEach((a) => a.classList.remove('active'));
      return;
    }
    const atBottom = innerHeight + scrollY >= document.documentElement.scrollHeight - 12;
    let activeSection = atBottom ? sections[sections.length - 1] : null;
    if (!activeSection) {
      const triggerY = headerH + 64;
      for (const s of sections) {
        if (s.getBoundingClientRect().top <= triggerY) {
          activeSection = s;
        }
      }
    }
    const activeHref = activeSection ? '#' + activeSection.id : '';
    navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === activeHref));
  }

  addEventListener(
    'scroll',
    () => {
      if (!spyTicking) {
        spyTicking = true;
        requestAnimationFrame(updateActiveNav);
      }
    },
    { passive: true }
  );
  updateActiveNav();
})();
