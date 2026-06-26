/* Mortgage Risk Under Climate Stress — dashboard interactions */
(function () {
  'use strict';
  var DASH = window.DASH || { series: [], stats: {} };
  var S = DASH.series, ST = DASH.stats;
  var CR = '#A51C30', CRsoft = '#C8102E', GOLD = '#B79A5B', INK = '#1A1718';

  var labels = S.map(function (d) { return d.date; });
  var dq = S.map(function (d) { return d.dq; });
  var temp = S.map(function (d) { return d.temp; });
  var precip = S.map(function (d) { return d.precip; });
  var dti = S.map(function (d) { return d.dti; });

  function norm(a) {
    var v = a.filter(function (x) { return x != null; });
    var mn = Math.min.apply(null, v), mx = Math.max.apply(null, v);
    return a.map(function (x) { return x == null ? null : (x - mn) / (mx - mn); });
  }

  // ---- Chart.js global look ----
  if (window.Chart) {
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#4A4446';
  }

  // year-only x ticks
  var xTicks = {
    callback: function (val, i) {
      var d = labels[i]; if (!d) return '';
      return d.slice(5) === '01' ? d.slice(0, 4) : '';
    },
    maxRotation: 0, autoSkip: false, font: { family: "'JetBrains Mono', monospace", size: 10 }
  };

  // plugin: shade disaster months
  var disasterBands = {
    id: 'disasterBands',
    beforeDatasetsDraw: function (c) {
      var xa = c.scales.x, ya = c.chartArea; if (!xa) return;
      var ctx = c.ctx; ctx.save(); ctx.fillStyle = 'rgba(165,28,48,0.07)';
      for (var i = 0; i < S.length; i++) {
        if (S[i].disaster === 1) {
          var x0 = xa.getPixelForValue(i) - (xa.width / S.length) / 2;
          ctx.fillRect(x0, ya.top, xa.width / S.length, ya.bottom - ya.top);
        }
      }
      ctx.restore();
    }
  };

  function grad(ctx, area, hex) {
    if (!area) return hex;
    var g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
    g.addColorStop(0, hex + 'cc'); g.addColorStop(1, hex + '05'); return g;
  }
  var baseLine = { tension: 0.32, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 };
  var noGrid = { grid: { display: false }, border: { display: false } };
  var softGrid = { grid: { color: 'rgba(26,23,24,0.06)' }, border: { display: false } };

  // ---- Hero backdrop ----
  var heroEl = document.getElementById('heroChart');
  if (heroEl && window.Chart) {
    new Chart(heroEl, {
      type: 'line',
      data: { labels: labels, datasets: [{
        data: dq, borderColor: CR, borderWidth: 2, fill: true, tension: 0.35, pointRadius: 0,
        backgroundColor: function (c) { return grad(c.chart.ctx, c.chart.chartArea, '#A51C30'); }
      }] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 1400 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } } }
    });
  }

  // ---- Timeline ----
  var tEl = document.getElementById('dqChart');
  var tChart;
  if (tEl && window.Chart) {
    tChart = new Chart(tEl, {
      type: 'line',
      data: { labels: labels, datasets: [
        { label: '60+ day delinquency (%)', data: dq, borderColor: CR, fill: true,
          backgroundColor: function (c) { return grad(c.chart.ctx, c.chart.chartArea, '#A51C30'); }, ...baseLine },
        { label: 'Origination DTI (%)', data: dti, borderColor: GOLD, borderDash: [5, 4],
          fill: false, hidden: true, ...baseLine }
      ] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false },
          tooltip: { backgroundColor: INK, padding: 12, titleFont: { family: "'JetBrains Mono',monospace" },
            callbacks: { afterTitle: function (it) { return S[it[0].dataIndex].disaster ? '⚠ disaster month' : ''; } } } },
        scales: { x: xTicks && Object.assign({}, noGrid, { ticks: xTicks }),
          y: Object.assign({ beginAtZero: true, ticks: { callback: function (v) { return v + '%'; } } }, softGrid) } },
      plugins: [disasterBands]
    });
    document.querySelectorAll('[data-tl]').forEach(function (b) {
      b.addEventListener('click', function () {
        var i = +b.dataset.tl; var meta = tChart.getDatasetMeta(i);
        meta.hidden = !(meta.hidden === null ? tChart.data.datasets[i].hidden : meta.hidden);
        b.classList.toggle('on'); tChart.update();
      });
    });
  }

  // ---- Climate overlay (normalized) ----
  var cEl = document.getElementById('climateChart');
  var cChart;
  if (cEl && window.Chart) {
    cChart = new Chart(cEl, {
      type: 'line',
      data: { labels: labels, datasets: [
        { label: 'Delinquency', data: norm(dq), borderColor: CR, ...baseLine, borderWidth: 2.4 },
        { label: 'Temperature', data: norm(temp), borderColor: GOLD, ...baseLine },
        { label: 'Precipitation', data: norm(precip), borderColor: '#3A6EA5', ...baseLine, hidden: true }
      ] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false },
          tooltip: { backgroundColor: INK, padding: 12 } },
        scales: { x: Object.assign({}, noGrid, { ticks: xTicks }),
          y: Object.assign({ min: 0, max: 1, ticks: { display: false } }, softGrid) } }
    });
    document.querySelectorAll('[data-cl]').forEach(function (b) {
      b.addEventListener('click', function () {
        var i = +b.dataset.cl; var m = cChart.getDatasetMeta(i);
        m.hidden = !(m.hidden === null ? cChart.data.datasets[i].hidden : m.hidden);
        b.classList.toggle('on'); cChart.update();
      });
    });
  }

  // ---- Count-up stats ----
  function countUp(el) {
    var target = parseFloat(el.dataset.count), dec = (el.dataset.dec ? +el.dataset.dec : 0), t0 = null, dur = 1300;
    function step(ts) {
      if (!t0) t0 = ts; var p = Math.min((ts - t0) / dur, 1);
      var e = 1 - Math.pow(1 - p, 3);
      el.firstChild.nodeValue = (target * e).toFixed(dec);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---- Reveal + trigger count-up ----
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) {
        en.target.classList.add('in');
        en.target.querySelectorAll && en.target.querySelectorAll('[data-count]').forEach(countUp);
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.18 });
  document.querySelectorAll('.reveal, .stats').forEach(function (n) { io.observe(n); });
})();
