(() => {
  const $ = (id) => document.getElementById(id);

  const taskSelect = $("taskSelect");
  const episodeInput = $("episodeInput");
  const posLabel = $("posLabel");
  const prevBtn = $("prevBtn");
  const nextBtn = $("nextBtn");
  const outcomeTag = $("outcomeTag");
  const statusBadge = $("statusBadge");
  const calcBtn = $("calcBtn");
  const themeToggleBtn = $("themeToggleBtn");
  const toast = $("toast");

  const GROUPS = {
    left: { labels: [0, 1, 2, 3, 4, 5, 6].map((j) => `lj${j}`), hasAction: true, legendEl: $("legendLeft"), svg: document.querySelector('.jointChart[data-group="left"]'), diagEl: $("leftDiag") },
    right: { labels: [0, 1, 2, 3, 4, 5, 6].map((j) => `rj${j}`), hasAction: true, legendEl: $("legendRight"), svg: document.querySelector('.jointChart[data-group="right"]'), diagEl: $("rightDiag") },
    lift: { labels: ["lift"], hasAction: false, legendEl: $("legendLift"), svg: document.querySelector('.jointChart[data-group="lift"]') },
  };

  const SERIES_COLOR_VAR = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7"];

  const videoEls = { "observation.images.zed": $("cam-zed"), "observation.images.fish0": $("cam-fish0"), "observation.images.fish1": $("cam-fish1") };
  const masterVideo = videoEls["observation.images.zed"];
  const playBtn = $("playBtn");
  const scrubber = $("scrubber");
  const timeReadout = $("timeReadout");

  let episode = null; // current /api/episode payload
  let visible = {}; // label -> bool
  let playheadTime = 0;
  let scrubbing = false;
  let currentTaskIndex = null; // task_index of the task dropdown's current selection
  let currentTaskEpisodes = []; // [{position, episode_index, length, success}], sorted by episode_index
  let suppressTaskSelectChange = false; // true while we set taskSelect.value programmatically

  function seriesColor(label, groupKey) {
    if (groupKey === "lift") return getVar("--series-lift");
    const idx = parseInt(label.slice(2), 10); // "lj3" -> 3, "rj3" -> 3
    return getVar(SERIES_COLOR_VAR[idx % SERIES_COLOR_VAR.length]);
  }

  function getVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function showToast(msg, isError) {
    toast.textContent = msg;
    toast.className = isError ? "show error" : "show";
    setTimeout(() => (toast.className = ""), 2600);
  }

  // ---------- data loading ----------

  async function fetchTasks() {
    const r = await fetch("/api/tasks");
    return r.json();
  }

  async function fetchTaskEpisodes(taskIndex) {
    const r = await fetch(`/api/tasks/${taskIndex}/episodes`);
    if (!r.ok) throw new Error(`task ${taskIndex} episodes: ${r.status}`);
    return r.json();
  }

  async function fetchEpisode(idx) {
    const r = await fetch(`/api/episode/${idx}`);
    if (!r.ok) throw new Error(`episode ${idx}: ${r.status}`);
    return r.json();
  }

  async function calculateActionQ(idx) {
    const r = await fetch(`/api/episode/${idx}/action-q`, { method: "POST" });
    if (!r.ok) throw new Error(`calculate action-q ${idx}: ${r.status}`);
    return r.json();
  }

  // ---------- rendering ----------

  function resetVisibility() {
    visible = {};
    for (const g of Object.values(GROUPS)) {
      for (const label of g.labels) visible[label] = true;
    }
  }

  function renderLegend(groupKey) {
    const g = GROUPS[groupKey];
    g.legendEl.innerHTML = "";
    for (const label of g.labels) {
      const chip = document.createElement("div");
      chip.className = "legend-chip" + (visible[label] ? "" : " off");
      chip.dataset.label = label;
      const swatch = document.createElement("span");
      swatch.className = "swatch line-swatch";
      swatch.style.borderTopColor = seriesColor(label, groupKey);
      chip.appendChild(swatch);
      const text = document.createElement("span");
      text.textContent = label;
      chip.appendChild(text);
      chip.addEventListener("click", () => {
        visible[label] = !visible[label];
        chip.classList.toggle("off", !visible[label]);
        renderChart(groupKey);
      });
      g.legendEl.appendChild(chip);
    }
  }

  const PAD_L = 46, PAD_R = 14, PAD_T = 10, PAD_B = 26;

  function renderChart(groupKey) {
    const g = GROUPS[groupKey];
    const svg = g.svg;
    if (!episode) {
      svg.innerHTML = "";
      return;
    }
    const vb = svg.viewBox.baseVal;
    const W = vb.width || 1000;
    const H = vb.height || 260;
    const t = episode.timestamp;
    const n = t.length;
    if (n === 0) {
      svg.innerHTML = "";
      return;
    }

    // gather visible series (obs always available; action only if computed)
    const series = [];
    for (const label of g.labels) {
      if (!visible[label]) continue;
      const color = seriesColor(label, groupKey);
      series.push({ label, color, data: episode.obs[label], dashed: false });
      if (g.hasAction && episode.action_q && episode.action_q[label]) {
        series.push({ label, color, data: episode.action_q[label], dashed: true });
      }
    }

    // y-domain from ALL series in the group (not just visible) so toggling doesn't rescale
    let yMin = Infinity, yMax = -Infinity;
    for (const label of g.labels) {
      const obsArr = episode.obs[label];
      for (let i = 0; i < obsArr.length; i++) {
        if (obsArr[i] < yMin) yMin = obsArr[i];
        if (obsArr[i] > yMax) yMax = obsArr[i];
      }
      if (g.hasAction && episode.action_q && episode.action_q[label]) {
        const actArr = episode.action_q[label];
        for (let i = 0; i < actArr.length; i++) {
          if (actArr[i] < yMin) yMin = actArr[i];
          if (actArr[i] > yMax) yMax = actArr[i];
        }
      }
    }
    if (!isFinite(yMin) || !isFinite(yMax)) { yMin = 0; yMax = 1; }
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
    const pad = (yMax - yMin) * 0.08;
    yMin -= pad; yMax += pad;

    const tMin = t[0], tMax = t[n - 1] || t[0] + 1;
    const xOf = (ts) => PAD_L + ((ts - tMin) / (tMax - tMin || 1)) * (W - PAD_L - PAD_R);
    const yOf = (v) => PAD_T + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - PAD_T - PAD_B);
    g.xOf = xOf;
    g.tOf = (px) => tMin + ((px - PAD_L) / (W - PAD_L - PAD_R)) * (tMax - tMin);
    g.tMin = tMin;
    g.tMax = tMax;

    const parts = [];

    // grid + y labels
    const GRID_N = 4;
    for (let i = 0; i <= GRID_N; i++) {
      const v = yMin + ((yMax - yMin) * i) / GRID_N;
      const y = yOf(v);
      parts.push(`<line class="chart-grid" x1="${PAD_L}" y1="${y.toFixed(1)}" x2="${W - PAD_R}" y2="${y.toFixed(1)}" />`);
      parts.push(`<text class="chart-axis-label" x="${PAD_L - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`);
    }
    // x labels (5 ticks)
    const X_N = 5;
    for (let i = 0; i <= X_N; i++) {
      const ts = tMin + ((tMax - tMin) * i) / X_N;
      const x = xOf(ts);
      parts.push(`<text class="chart-axis-label" x="${x.toFixed(1)}" y="${H - PAD_B + 16}" text-anchor="middle">${ts.toFixed(1)}s</text>`);
    }
    parts.push(`<line class="chart-axis" x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${H - PAD_B}" />`);
    parts.push(`<line class="chart-axis" x1="${PAD_L}" y1="${H - PAD_B}" x2="${W - PAD_R}" y2="${H - PAD_B}" />`);

    // downsample for very long episodes to keep the polyline light
    const STEP = Math.max(1, Math.floor(n / 1500));

    for (const s of series) {
      const pts = [];
      for (let i = 0; i < n; i += STEP) {
        pts.push(`${xOf(t[i]).toFixed(1)},${yOf(s.data[i]).toFixed(1)}`);
      }
      if ((n - 1) % STEP !== 0) pts.push(`${xOf(t[n - 1]).toFixed(1)},${yOf(s.data[n - 1]).toFixed(1)}`);
      parts.push(
        `<polyline class="chart-line${s.dashed ? " action-line" : ""}" points="${pts.join(" ")}" stroke="${s.color}" />`
      );
    }

    const px = xOf(Math.max(tMin, Math.min(tMax, playheadTime)));
    parts.push(`<line class="chart-playhead" x1="${px.toFixed(1)}" y1="${PAD_T}" x2="${px.toFixed(1)}" y2="${H - PAD_B}" />`);

    svg.innerHTML = parts.join("");
  }

  function updatePlayhead(t) {
    playheadTime = t;
    for (const g of Object.values(GROUPS)) {
      if (!g.xOf) continue;
      const el = g.svg.querySelector(".chart-playhead");
      if (!el) continue;
      const x = g.xOf(Math.max(g.tMin, Math.min(g.tMax, t)));
      el.setAttribute("x1", x.toFixed(1));
      el.setAttribute("x2", x.toFixed(1));
    }
  }

  function formatTime(t) {
    if (!isFinite(t)) t = 0;
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
  }

  function loadVideos(urls) {
    for (const [key, el] of Object.entries(videoEls)) {
      if (urls && urls[key]) {
        el.src = urls[key];
      } else {
        el.removeAttribute("src");
      }
      el.load();
    }
    scrubber.value = 0;
    playBtn.textContent = "▶";
  }

  function seekTo(t) {
    const dur = episode ? episode.timestamp[episode.timestamp.length - 1] : 0;
    const clamped = Math.max(0, Math.min(t, dur || t));
    for (const el of Object.values(videoEls)) {
      if (el.src) el.currentTime = clamped;
    }
    scrubber.value = clamped;
    timeReadout.textContent = `${formatTime(clamped)} / ${formatTime(dur)}`;
    updatePlayhead(clamped);
  }

  masterVideo.addEventListener("timeupdate", () => {
    if (scrubbing) return;
    const t = masterVideo.currentTime;
    for (const [key, el] of Object.entries(videoEls)) {
      if (key === "observation.images.zed" || !el.src) continue;
      if (Math.abs(el.currentTime - t) > 0.15) el.currentTime = t;
    }
    const dur = episode ? episode.timestamp[episode.timestamp.length - 1] : masterVideo.duration || 0;
    scrubber.value = t;
    timeReadout.textContent = `${formatTime(t)} / ${formatTime(dur)}`;
    updatePlayhead(t);
  });

  masterVideo.addEventListener("ended", () => {
    playBtn.textContent = "▶";
    for (const el of Object.values(videoEls)) el.pause();
  });

  playBtn.addEventListener("click", () => {
    if (!episode) return;
    if (masterVideo.paused) {
      for (const el of Object.values(videoEls)) if (el.src) el.play().catch(() => {});
      playBtn.textContent = "⏸";
    } else {
      for (const el of Object.values(videoEls)) el.pause();
      playBtn.textContent = "▶";
    }
  });

  scrubber.addEventListener("input", () => {
    scrubbing = true;
    for (const el of Object.values(videoEls)) el.pause();
    playBtn.textContent = "▶";
    seekTo(parseFloat(scrubber.value));
  });
  scrubber.addEventListener("change", () => { scrubbing = false; });

  function attachChartSeek(groupKey) {
    const g = GROUPS[groupKey];
    g.svg.addEventListener("pointerdown", (e) => {
      if (!episode || !g.tOf) return;
      const rect = g.svg.getBoundingClientRect();
      const vb = g.svg.viewBox.baseVal;
      const px = ((e.clientX - rect.left) / rect.width) * vb.width;
      const t = g.tOf(px);
      for (const el of Object.values(videoEls)) el.pause();
      playBtn.textContent = "▶";
      seekTo(t);
    });
  }

  function renderAllCharts() {
    for (const key of Object.keys(GROUPS)) renderChart(key);
  }

  function renderDiagnostics() {
    for (const arm of ["left", "right"]) {
      const el = GROUPS[arm].diagEl;
      if (!el) continue;
      if (!episode || !episode.action_q || !episode.action_q_diagnostics) {
        el.textContent = "";
        continue;
      }
      const d = episode.action_q_diagnostics[arm];
      if (!d) { el.textContent = ""; continue; }
      el.textContent =
        `solved ${(d.solved_rate * 100).toFixed(1)}% · pos err mean/max ${d.pos_err_mean.toExponential(1)}/${d.pos_err_max.toExponential(1)} m · ` +
        `rot err mean/max ${d.rot_err_mean.toExponential(1)}/${d.rot_err_max.toExponential(1)} rad`;
    }
  }

  // ---------- episode navigation ----------

  function updateTopbar() {
    if (!episode) return;
    posLabel.textContent = `${episode.position} / ${episode.total_in_task}`;
    episodeInput.value = episode.position;
    episodeInput.placeholder = String(episode.total_in_task);
    episodeInput.max = episode.total_in_task;

    suppressTaskSelectChange = true;
    taskSelect.value = String(episode.task_index);
    suppressTaskSelectChange = false;

    const success = episode.success;
    outcomeTag.textContent = String(success);
    outcomeTag.className = "outcome-tag" + (success === "success" || success === true ? " ok" : success === "fail" || success === false ? " fail" : "");
    prevBtn.disabled = episode.position <= 1;
    nextBtn.disabled = episode.position >= episode.total_in_task;
  }

  async function ensureTaskEpisodes(taskIndex) {
    if (currentTaskIndex === taskIndex && currentTaskEpisodes.length) return currentTaskEpisodes;
    currentTaskEpisodes = await fetchTaskEpisodes(taskIndex);
    currentTaskIndex = taskIndex;
    return currentTaskEpisodes;
  }

  function setStatus(kind, text) {
    statusBadge.className = "status-badge" + (kind ? ` ${kind}` : "");
    statusBadge.textContent = text;
  }

  async function loadEpisode(idx) {
    setStatus("", "loading…");
    calcBtn.disabled = true;
    try {
      const data = await fetchEpisode(idx);
      episode = data;
      episode.action_q_diagnostics = data.action_q ? data.action_q.diagnostics : null;
      if (data.action_q) episode.action_q = data.action_q.qtarget;
      await ensureTaskEpisodes(episode.task_index);
      resetVisibility();
      playheadTime = 0;
      scrubber.max = episode.timestamp[episode.timestamp.length - 1] || 1;
      loadVideos(episode.videos);
      timeReadout.textContent = `0:00.0 / ${formatTime(episode.timestamp[episode.timestamp.length - 1] || 0)}`;
      updateTopbar();
      renderLegend("left");
      renderLegend("right");
      renderLegend("lift");
      renderAllCharts();
      renderDiagnostics();
      setStatus(episode.action_q ? "done" : "", episode.action_q ? "action q computed" : "not computed");
    } catch (e) {
      setStatus("error", "load failed");
      showToast(String(e), true);
    } finally {
      calcBtn.disabled = false;
    }
  }

  async function onCalculate() {
    if (!episode) return;
    setStatus("computing", "computing…");
    calcBtn.disabled = true;
    try {
      const result = await calculateActionQ(episode.episode_index);
      episode.action_q = result.qtarget;
      episode.action_q_diagnostics = result.diagnostics;
      renderAllCharts();
      renderDiagnostics();
      setStatus("done", "action q computed");
    } catch (e) {
      setStatus("error", "compute failed");
      showToast(String(e), true);
    } finally {
      calcBtn.disabled = false;
    }
  }

  function goToPosition(pos) {
    if (!currentTaskEpisodes.length) return;
    const clamped = Math.max(1, Math.min(currentTaskEpisodes.length, pos));
    loadEpisode(currentTaskEpisodes[clamped - 1].episode_index);
  }

  prevBtn.addEventListener("click", () => episode && goToPosition(episode.position - 1));
  nextBtn.addEventListener("click", () => episode && goToPosition(episode.position + 1));
  episodeInput.addEventListener("change", () => {
    const v = parseInt(episodeInput.value, 10);
    if (!isNaN(v)) goToPosition(v);
  });
  taskSelect.addEventListener("change", () => {
    if (suppressTaskSelectChange || taskSelect.value === "") return;
    const t_idx = Number(taskSelect.value);
    ensureTaskEpisodes(t_idx).then(() => goToPosition(1));
  });
  calcBtn.addEventListener("click", onCalculate);

  themeToggleBtn.addEventListener("click", () => {
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    if (isLight) {
      document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      localStorage.setItem("theme", "light");
    }
    renderAllCharts();
  });

  window.addEventListener("resize", renderAllCharts);

  attachChartSeek("left");
  attachChartSeek("right");
  attachChartSeek("lift");

  function populateTaskSelect(tasks) {
    taskSelect.innerHTML = "";
    for (const t of tasks) {
      const opt = document.createElement("option");
      opt.value = t.task_index;
      opt.textContent = `${t.task} (${t.total_episodes})`;
      taskSelect.appendChild(opt);
    }
  }

  (async function init() {
    try {
      const tasks = await fetchTasks();
      populateTaskSelect(tasks);
      await loadEpisode(0);
    } catch (e) {
      setStatus("error", "init failed");
      showToast(String(e), true);
    }
  })();
})();
