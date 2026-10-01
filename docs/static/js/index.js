/* BudgetMem project page. No runtime dependencies or build step. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  // Table 1, arXiv:2602.06025v3 (May 2026), performance-first setting.
  // Each triple is [F1 (%), LLM-Judge (%), reported token-based cost (USD)].
  // Dataset order: LoCoMo, LongMemEval, HotpotQA. No interpolated measurements.
  const methods = [
    'ReadAgent', 'MemoryBank', 'A-MEM', 'LangMem', 'Mem0',
    'MemoryOS', 'LightMem', 'BudgetMem–IMP', 'BudgetMem–REA', 'BudgetMem–CAP'
  ];
  const scores = {
    llama: [
      [[22.48,31.05,.57],[20.75,27.72,13.68],[15.33,30.08,4.19]],
      [[22.27,28.98,.73],[26.74,32.67,3.94],[22.25,23.75,7.75]],
      [[26.43,32.96,2.88],[21.74,33.17,80.02],[43.25,54.69,26.74]],
      [[22.31,25.96,.48],[12.00,17.00,16.60],[22.78,22.66,10.95]],
      [[11.04,28.18,2.89],[27.70,42.08,13.57],[28.03,36.72,4.30]],
      [[30.62,34.55,1.97],[12.97,33.50,38.83],[34.50,43.36,13.32]],
      [[33.88,40.76,1.50],[26.74,48.51,5.28],[45.73,58.37,10.10]],
      [[38.75,50.32,1.80],[37.47,56.00,.71],[49.31,65.77,1.35]],
      [[40.92,52.23,2.90],[40.53,58.00,.67],[51.12,61.93,.99]],
      [[43.05,54.62,2.40],[40.24,60.50,.80],[53.87,64.85,.93]]
    ],
    qwen: [
      [[22.45,31.37,.24],[27.72,20.75,4.97],[18.05,25.78,1.75]],
      [[23.53,34.71,.25],[10.56,28.22,3.45],[18.64,31.25,1.79]],
      [[27.65,38.54,2.88],[10.82,31.19,21.00],[40.54,50.39,8.34]],
      [[20.89,23.40,.14],[11.01,14.00,3.99],[20.77,21.09,17.56]],
      [[10.77,25.32,1.15],[25.71,36.14,4.96],[24.72,37.89,2.02]],
      [[35.43,38.85,.75],[13.35,33.00,15.84],[41.21,53.52,11.68]],
      [[32.85,42.83,.70],[27.70,47.52,3.39],[41.29,55.42,8.56]],
      [[40.14,54.38,.80],[29.18,52.00,.30],[46.67,57.42,.63]],
      [[40.19,53.34,1.11],[35.84,59.00,.26],[57.67,70.83,.17]],
      [[41.22,53.18,.61],[31.01,56.00,.17],[58.70,72.08,.22]]
    ]
  };
  const benchmarks = {
    locomo: { name: 'LoCoMo', index: 0 },
    longmemeval: { name: 'LongMemEval', index: 1 },
    hotpotqa: { name: 'HotpotQA', index: 2 }
  };
  const backbones = {
    llama: 'LLaMA-3.3-70B-Instruct',
    qwen: 'Qwen3-Next-80B-A3B-Instruct'
  };
  let selectedBenchmark = 'longmemeval';
  const modelSelect = $('#backbone-select');

  function renderResults(announce = true) {
    const benchmark = benchmarks[selectedBenchmark];
    const model = modelSelect.value;
    const rows = methods.map((name, index) => ({ name, ours: index >= 7, values: scores[model][index][benchmark.index] }));
    const baseline = rows.slice(0, 7).reduce((best, row) => row.values[1] > best.values[1] ? row : best);
    const variants = rows.slice(7);
    const winner = variants.reduce((best, row) => row.values[1] > best.values[1] ? row : best);
    // Relative Judge improvement over the strongest non-BudgetMem baseline
    // for this benchmark and backbone, computed from the reported scores.
    const relativeGain = (winner.values[1] - baseline.values[1]) / baseline.values[1] * 100;
    const shown = [baseline, ...variants];

    $('#chart-title').textContent = benchmark.name;
    $('#chart-subtitle').textContent = `${backbones[model]} · performance-first setting`;
    $('#benchmark-panel').setAttribute('aria-labelledby', `tab-${selectedBenchmark}`);
    $$('.bar-row').forEach((element, index) => {
      const row = shown[index];
      element.classList.toggle('best', row === winner);
      $('.bar-name', element).textContent = row.name;
      $('.bar-fill', element).style.setProperty('--score', `${row.values[1]}%`);
      $('.bar-value', element).textContent = row.values[1].toFixed(2);
    });
    $('#bar-chart').setAttribute('aria-label', `${benchmark.name} Judge scores: ${shown.map(row => `${row.name} ${row.values[1].toFixed(2)}`).join(', ')}.`);
    $('#highlight-method').textContent = winner.name;
    $('#highlight-score').textContent = winner.values[1].toFixed(2);
    $('#highlight-gain').textContent = `+${relativeGain.toFixed(2)}% relative to ${baseline.name}`;
    $('#highlight-f1').textContent = winner.values[0].toFixed(2);
    $('#highlight-cost').textContent = `$${winner.values[2].toFixed(2)}`;
    $('#results-table-caption').textContent = `${benchmark.name} · ${backbones[model]} · performance-first (λ = 0)`;
    const bestValues = [
      Math.max(...rows.map(row => row.values[0])),
      Math.max(...rows.map(row => row.values[1])),
      Math.min(...rows.map(row => row.values[2]))
    ];
    const fragment = document.createDocumentFragment();
    rows.forEach(row => {
      const tr = document.createElement('tr');
      if (row.ours) tr.className = 'ours';
      const th = document.createElement('th');
      th.scope = 'row';
      th.textContent = row.name;
      tr.append(th);
      row.values.forEach((value, index) => {
        const td = document.createElement('td');
        if (value === bestValues[index]) {
          const strong = document.createElement('strong');
          strong.textContent = value.toFixed(2);
          td.append(strong);
        } else td.textContent = value.toFixed(2);
        tr.append(td);
      });
      fragment.append(tr);
    });
    $('#results-table-body').replaceChildren(fragment);
    if (announce) $('#results-announcement').textContent = `${benchmark.name}, ${backbones[model]}. Best Judge: ${winner.name}, ${winner.values[1].toFixed(2)}. Relative Judge improvement over ${baseline.name}: ${relativeGain.toFixed(2)} percent. Reported cost: ${winner.values[2].toFixed(2)} US dollars.`;
  }

  // WAI-ARIA tabs: automatic activation, roving focus, arrows, Home and End.
  function setupTabs(tablist, onSelect) {
    const tabs = $$('[role="tab"]', tablist);
    function select(tab) {
      tabs.forEach(item => {
        const active = item === tab;
        item.setAttribute('aria-selected', String(active));
        item.tabIndex = active ? 0 : -1;
      });
      onSelect(tab);
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          select(tabs[next]);
          tabs[next].focus();
        }
      });
    });
    tablist.hidden = false;
  }
  setupTabs($('#benchmark-tabs'), tab => {
    selectedBenchmark = tab.dataset.benchmark;
    renderResults();
  });
  modelSelect.addEventListener('change', () => renderResults());
  renderResults(false);
  $('.results-controls').hidden = false;

  const analysisPanels = $$('.analysis-panel');
  function showAnalysis(tab) {
    analysisPanels.forEach(panel => { panel.hidden = panel.id !== tab.getAttribute('aria-controls'); });
  }
  setupTabs($('.analysis-tabs'), showAnalysis);
  showAnalysis($('.analysis-tabs [aria-selected="true"]'));

  // An explicitly illustrative allocation, not an inference or measured result.
  const allocations = [[0, 0, 1, 0, 0], [1, 2, 1, 0, 2], [2, 2, 2, 1, 2]];
  const preferences = ['Cost-focused', 'Balanced', 'Performance-focused'];
  const tierNames = ['Low', 'Mid', 'High'];
  const moduleNames = ['Filter', 'Entity', 'Temporal', 'Topic', 'Summary'];
  const range = $('#budget-range');
  const columns = $$('.tier-column');
  function drawRoute() {
    const width = $('.tier-columns').clientWidth;
    if (!width) return;
    const allocation = allocations[Number(range.value)];
    const points = columns.map((column, index) => {
      const x = (column.offsetLeft + column.offsetWidth / 2) / width * 500;
      const y = (2 - allocation[index]) * 58 + 29;
      return `${x.toFixed(2)},${y}`;
    });
    $('#route-path').setAttribute('points', points.join(' '));
  }
  function updateBudget() {
    const index = Number(range.value);
    columns.forEach((column, moduleIndex) => {
      $$('.tier-cell', column).forEach(cell => cell.classList.toggle('selected', Number(cell.dataset.tier) === allocations[index][moduleIndex]));
    });
    $('#budget-value').textContent = preferences[index];
    range.setAttribute('aria-valuetext', preferences[index]);
    $('#routing-description').textContent = `${preferences[index]} example: ${moduleNames.map((name, moduleIndex) => `${name} ${tierNames[allocations[index][moduleIndex]]}`).join(', ')}.`;
    drawRoute();
  }
  range.addEventListener('input', updateBudget);
  $('.budget-control').hidden = false;
  updateBudget();
  if ('ResizeObserver' in window) new ResizeObserver(drawRoute).observe($('.tier-columns'));
  else window.addEventListener('resize', drawRoute);

  const menuButton = $('.menu-toggle');
  const nav = $('#site-nav');
  function closeMenu(restoreFocus = false) {
    nav.classList.remove('is-open');
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-label', 'Open navigation');
    if (restoreFocus) menuButton.focus();
  }
  menuButton.hidden = false;
  menuButton.addEventListener('click', () => {
    const open = menuButton.getAttribute('aria-expanded') !== 'true';
    nav.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  });
  $$('a', nav).forEach(link => link.addEventListener('click', () => closeMenu()));
  document.addEventListener('click', event => {
    if (!$('.site-header').contains(event.target)) closeMenu();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') closeMenu(true);
  });
  const mobileMedia = window.matchMedia('(max-width: 700px)');
  mobileMedia.addEventListener('change', () => closeMenu());

  // Native dialog keeps keyboard focus in the figure and supports Escape.
  const dialog = $('#figure-dialog');
  const zoomButton = $('#dialog-zoom');
  let figureTrigger;
  function setFigureZoom(zoomed) {
    dialog.classList.toggle('is-zoomed', zoomed);
    zoomButton.setAttribute('aria-pressed', String(zoomed));
    zoomButton.textContent = zoomed ? 'Fit figure' : 'Zoom in';
  }
  if (typeof dialog.showModal === 'function') {
    $$('[data-lightbox]').forEach(link => link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      figureTrigger = link;
      const caption = link.dataset.caption;
      $('#dialog-image').src = link.href;
      $('#dialog-image').alt = caption;
      $('#dialog-caption').textContent = caption;
      $('#dialog-original').href = link.href;
      setFigureZoom(false);
      dialog.showModal();
      document.body.classList.add('modal-open');
      $('.dialog-close').focus();
    }));
    $('.dialog-close').addEventListener('click', () => dialog.close());
    zoomButton.addEventListener('click', () => setFigureZoom(zoomButton.getAttribute('aria-pressed') !== 'true'));
    let backdropDown = false;
    dialog.addEventListener('pointerdown', event => { backdropDown = event.target === dialog; });
    dialog.addEventListener('click', event => {
      if (event.target === dialog && backdropDown) {
        const rect = dialog.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
      }
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('modal-open');
      figureTrigger?.focus({ preventScroll: true });
    });
  }

  const copyButton = $('#copy-citation');
  let copyTimer;
  function fallbackCopy(text) {
    const field = document.createElement('textarea');
    field.value = text;
    field.setAttribute('readonly', '');
    field.style.cssText = 'position:fixed;top:0;left:-9999px;opacity:0';
    document.body.append(field);
    field.select();
    let copied = false;
    try { copied = document.execCommand('copy'); } catch { /* Manual selection follows. */ }
    field.remove();
    copyButton.focus({ preventScroll: true });
    return copied;
  }
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    clearTimeout(copyTimer);
    const text = $('#citation-text').textContent.trim();
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch { /* file:// and restricted clipboard contexts use a local fallback. */ }
    if (!copied) copied = fallbackCopy(text);
    $('span', copyButton).textContent = copied ? 'Copied!' : 'Select & copy';
    $('#copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    if (!copied) {
      const selection = window.getSelection();
      const selectedText = document.createRange();
      selectedText.selectNodeContents($('#citation-text'));
      selection.removeAllRanges();
      selection.addRange(selectedText);
    }
    copyTimer = setTimeout(() => {
      $('span', copyButton).textContent = 'Copy BibTeX';
      $('#copy-status').textContent = '';
    }, 3500);
  });

  const navLinks = $$('a', nav);
  const sections = navLinks.map(link => $(link.getAttribute('href')));
  let framePending = false;
  function updateScroll() {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const progress = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
    $('.reading-progress').style.transform = `scaleX(${progress})`;
    let active = -1;
    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= 180) active = index;
    });
    navLinks.forEach((link, index) => {
      if (index === active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    framePending = false;
  }
  function scheduleScrollUpdate() {
    if (!framePending) { framePending = true; requestAnimationFrame(updateScroll); }
  }
  window.addEventListener('scroll', scheduleScrollUpdate, { passive: true });
  window.addEventListener('resize', scheduleScrollUpdate);
  window.addEventListener('load', updateScroll);
  updateScroll();
})();
